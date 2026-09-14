"""this file will handle db_state and temporary WAL(Write Ahead Log) file and manage compaction"""

import json
import logging
import os
import time
from threading import Lock

from .config import COMPACT_THRESHOLD, DB_FILE, WAL_FILE, WAL_SYNC_ON_WRITE

DEFAULT_DB = {
    "default": "default"
}  # the default entry we will use to populate database when no DB_FILE found
logger = logging.getLogger(__name__)


class DbEngine:
    def __init__(
        self,
        db_file=None,
        wal_file=None,
        compact_threshold=None,
        wal_sync_on_write=None,
    ):
        # instead of reading from module constants which is copied at import time, we can accept explicit values so tests/callers can pass inject isolated values
        self.db_file = db_file if db_file is not None else DB_FILE
        self.wal_file = wal_file if wal_file is not None else WAL_FILE
        self.compact_threshold = (
            compact_threshold if compact_threshold is not None else COMPACT_THRESHOLD
        )
        self.wal_sync_on_write = (
            wal_sync_on_write if wal_sync_on_write is not None else WAL_SYNC_ON_WRITE
        )
        # now validate the values cause we have to discard abnormal values given by caller
        if not self.db_file:
            raise ValueError("db_file must be a non-empty path")
        if not self.wal_file:
            raise ValueError("wal_file must be a non-empty path")
        if (
            isinstance(self.compact_threshold, bool)
            or not isinstance(self.compact_threshold, int)
            or self.compact_threshold < 1
        ):
            raise ValueError("compact_threshold must be a positive int")
        if not isinstance(self.wal_sync_on_write, bool):
            raise ValueError("wal_sync_on_write must be a bool")

        self.state = {}  # we create the db_state
        self.uncompacted_writes = 0  # number of entries that is loaded to memory and appended to WAL but not compacted yet
        self.lock = Lock()

    def boot(self):
        """first loads the db_file.Then checks if any wal_file with uncompacted changes is present.
        If yes then load those uncompacted changes to memory.then call compact to merge with db_file"""
        with self.lock:
            # step1: Check if DB_FILE exists.If not then populate it.
            if not os.path.isfile(self.db_file):
                with open(self.db_file, "w") as f:
                    json.dump(DEFAULT_DB, f)
                self.state = dict(DEFAULT_DB)
            # step2: If exists then try to load and if fails save a backup adn load fresh again
            else:
                try:
                    with open(self.db_file, "r") as f:
                        self.state = json.load(f)
                # if there is error we make a backup and start fresh
                except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
                    backup = f"{self.db_file}.corrupt-{int(time.time())}"
                    logger.warning(
                        "Corrupt database file %s (%s); moved to %s and starting fresh",
                        self.db_file,
                        e,
                        backup,
                    )
                    os.replace(
                        self.db_file, backup
                    )  # atomically replace the backup file with corrupted DB_FILE
                    self._fsync_dir(self.db_file)  # force fsync of parent directory

                    with open(self.db_file, "w") as f:
                        json.dump(DEFAULT_DB, f)
                    self.state = dict(DEFAULT_DB)

            # step3: Replay the WAL file if it's present and has contents
            if os.path.isfile(self.wal_file):
                with open(self.wal_file, "r") as f:
                    lines = f.readlines()
                for i, line in enumerate(lines):
                    # skip blank lines
                    if not line.strip():
                        continue

                    # handle corrupt/torn WAL line
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.warning(
                            "Truncated/corrupt WAL entry at line %d (%s); discarding it and %d entries after it",
                            i,
                            e,
                            len(lines) - i - 1,
                        )
                        break  # we stop at the torn line and discard everything after it
                    self.state.update(entry)
                self._compact_locked()  # call compact to merge with db_file

    def put(self, key: str, value: str):
        """updates memory.and also appends to the wal_file. and checks if compaction is needed.
        flush()+fsync() is used to make each PUT call crash safe.
        Without flush()+fsync(), calling f.write() only puts data into Python's own internal buffer.
        If the program crashes a millisecond later, that data vanishes before reaching the WAL file on disk.
        - flush() pushes the data out of Python's buffer via the actual write() syscall, into the OS's page cache.
        - fsync() then asks the OS to push its page cache to the disk, and waits for the disk to commit its own onboard cache to physical storage.
        This makes every single put() a bit slower, so it can be toggled via WAL_SYNC_ON_WRITE in config.py, to choose between maximum durability and high throughput.
        """
        with self.lock:
            # step1: Update memory with new key
            self.state[key] = value

            # setp2: Append to WAL file
            with open(self.wal_file, "a") as f:
                f.write(
                    json.dumps({key: value})
                    + "\n"  # using dumps instead of dump cause we are working with string.not file object
                )
                if self.wal_sync_on_write:
                    f.flush()  # push python buffer into OS
                    os.fsync(
                        f.fileno()
                    )  # force the OS to block execution until the bytes are stored into WAL file

            # step3: Increment the uncompacted_writes counter
            self.uncompacted_writes += 1

            # step4: Trigger compaction if needed
            if self.uncompacted_writes >= self.compact_threshold:
                self._compact_locked()

    def get(self, key: str):
        """looks up value of a key"""
        with self.lock:
            return self.state.get(key)

    # it is the locked version.when the caller function already acquires the self.lock,it should call this function instead of compact().calling compact() will cause deadlock.
    def _compact_locked(self):
        """writes current memory into db_file and clears the wal_file"""
        # NOTE:To prevent db_file corruption due to crash during the compaction process
        # we will first write to a temporary file,fsync() and then do a Atomic replace the old db_file with new one
        tmp_file = f"{self.db_file}.tmp"

        # load memory into the temp file
        with open(tmp_file, "w") as f:
            json.dump(self.state, f)
            f.flush()  # push python buffer into OS
            os.fsync(
                f.fileno()
            )  # force the OS to write it to physical disk.this is to make the operating system to actually write data from memory buffers to the physical disk, instead of leaving it sitting in a cache

        os.replace(tmp_file, self.db_file)  # atomic replace.safe
        self._fsync_dir(self.db_file)  # force fsync of parent directory

        open(self.wal_file, "w").close()  # trunicate the WAL to empty
        self.uncompacted_writes = 0  # set the uncompacted_write counter to zero

    # this function is called from the outside when the caller doesn't acquire the self.lock already
    def compact(self):
        with self.lock:
            self._compact_locked()

    def _fsync_dir(self, path):
        """this function will fsync dirctory containing `path`"""
        dir_path = os.path.dirname(os.path.abspath(path)) or "."
        fd = os.open(dir_path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)  # close fd even if fsync raises error
