"""Unit tests for DbEngine's WAL, compaction, crash-recovery, and validation logic."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from seriousdb.db_engine import DbEngine


class DbEngineWalAndCompactionTests(unittest.TestCase):
    def setUp(self):
        tmpdir = TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        self.db_file = str(Path(tmpdir.name) / ".sdb")
        self.wal_file = str(Path(tmpdir.name) / "changes.log")

    def test_boot_creates_default_db_when_missing(self):
        """If db_file doesn't exist yet, boot() should create it with the default seed entry."""
        db = DbEngine(db_file=self.db_file, wal_file=self.wal_file)
        db.boot()
        self.assertEqual(db.get("default"), "default")
        self.assertTrue(Path(self.db_file).is_file())

    def test_put_does_not_rewrite_db_file_before_threshold(self):
        """A single put() below the compaction threshold should be durable
        in the WAL, but db_file itself should be left untouched until
        compaction actually runs."""
        db = DbEngine(db_file=self.db_file, wal_file=self.wal_file, compact_threshold=50)
        db.boot()
        db.put("k1", "v1")

        on_disk = json.loads(Path(self.db_file).read_text())
        self.assertNotIn("k1", on_disk)  # not compacted yet
        wal_contents = Path(self.wal_file).read_text()
        self.assertIn('"k1": "v1"', wal_contents)  # but durable in the WAL

    def test_compaction_triggers_automatically_at_threshold(self):
        """Once uncompacted_writes reaches compact_threshold, put() should
        trigger compaction on its own -- db_file gets all entries and the
        WAL is truncated, without an explicit compact() call."""
        db = DbEngine(db_file=self.db_file, wal_file=self.wal_file, compact_threshold=5)
        db.boot()
        for i in range(5):
            db.put(f"k{i}", f"v{i}")

        self.assertEqual(Path(self.wal_file).read_text(), "")
        on_disk = json.loads(Path(self.db_file).read_text())
        for i in range(5):
            self.assertEqual(on_disk[f"k{i}"], f"v{i}")

    def test_boot_replays_uncompacted_wal_entries_after_simulated_crash(self):
        """Simulates a clean process restart with uncompacted writes still
        sitting in the WAL: a fresh DbEngine pointed at the same files
        should replay them into memory and compact, rather than losing
        them."""
        db1 = DbEngine(db_file=self.db_file, wal_file=self.wal_file, compact_threshold=50)
        db1.boot()
        db1.put("survives_crash", "yes")

        # simulate restart: fresh instance pointed at the same files
        db2 = DbEngine(db_file=self.db_file, wal_file=self.wal_file, compact_threshold=50)
        db2.boot()

        self.assertEqual(db2.get("survives_crash"), "yes")
        self.assertEqual(Path(self.wal_file).read_text(), "")

    def test_boot_quarantines_corrupt_db_file_instead_of_crashing(self):
        """If db_file itself is corrupt (unparseable JSON), boot() should
        back it up to a timestamped .corrupt-* file and start fresh with
        the default seed, rather than carshing."""
        Path(self.db_file).write_text("this is not valid json{{{")

        db = DbEngine(db_file=self.db_file, wal_file=self.wal_file)
        db.boot()  # should not raise

        self.assertEqual(db.get("default"), "default")
        corrupt_backups = list(
            Path(self.db_file).parent.glob(f"{Path(self.db_file).name}.corrupt-*")
        )
        self.assertEqual(len(corrupt_backups), 1)
        self.assertIn("not valid json", corrupt_backups[0].read_text())

    def test_manual_compact_flushes_and_clears_wal(self):
        """Calling compact() directly (as opposed to it being triggered
        automatically by put()) should persist all in-memory state to
        db_file and truncate the WAL."""
        db = DbEngine(db_file=self.db_file, wal_file=self.wal_file, compact_threshold=50)
        db.boot()
        db.put("a", "1")
        db.put("b", "2")

        db.compact()

        self.assertEqual(Path(self.wal_file).read_text(), "")
        on_disk = json.loads(Path(self.db_file).read_text())
        self.assertEqual(on_disk["a"], "1")
        self.assertEqual(on_disk["b"], "2")

    def test_boot_recovers_from_torn_final_wal_line(self):
        """Simulates a crash mid-append: the last WAL line is truncated JSON.
        boot() should keep everything before it and discard the torn line,
        rather than crashing."""
        db1 = DbEngine(db_file=self.db_file, wal_file=self.wal_file, compact_threshold=50)
        db1.boot()
        db1.put("good", "1")

        # simulate a crash mid-write: append a truncated JSON line directly,
        # bypassing put() so we don't trigger a real fsync'd write
        with open(self.wal_file, "a") as f:
            f.write('{"partial": "val')  # no closing brace, no trailing newline

        db2 = DbEngine(db_file=self.db_file, wal_file=self.wal_file, compact_threshold=50)
        db2.boot()  # must not raise

        self.assertEqual(db2.get("good"), "1")
        self.assertIsNone(db2.get("partial"))
        # boot() compacts after replay, so the torn WAL should be cleared too
        self.assertEqual(Path(self.wal_file).read_text(), "")

    def test_boot_discards_entries_after_a_torn_wal_line(self):
        """A bad line stops replay entirely -- anything written after it in the
        file is also discarded, since a torn write should only ever be the
        last line under normal (append-only) operation."""
        db1 = DbEngine(db_file=self.db_file, wal_file=self.wal_file, compact_threshold=50)
        db1.boot()
        db1.put("before", "1")

        with open(self.wal_file, "a") as f:
            f.write('not valid json at all\n')
            f.write(json.dumps({"after": "2"}) + "\n")

        db2 = DbEngine(db_file=self.db_file, wal_file=self.wal_file, compact_threshold=50)
        with self.assertLogs("seriousdb.db_engine", level="WARNING") as wrn:
            db2.boot()
        print(wrn.output) #print the warning just to be sure

        self.assertEqual(db2.get("before"), "1")
        self.assertIsNone(db2.get("after"))


class DbEngineValidationTests(unittest.TestCase):
    def setUp(self):
        tmpdir = TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        self.db_file = str(Path(tmpdir.name) / ".sdb")
        self.wal_file = str(Path(tmpdir.name) / "changes.log")

    def test_rejects_zero_compact_threshold(self):
        with self.assertRaises(ValueError):
            DbEngine(db_file=self.db_file, wal_file=self.wal_file, compact_threshold=0)

    def test_rejects_negative_compact_threshold(self):
        with self.assertRaises(ValueError):
            DbEngine(db_file=self.db_file, wal_file=self.wal_file, compact_threshold=-5)

    def test_rejects_bool_compact_threshold(self):
        with self.assertRaises(ValueError):
            DbEngine(db_file=self.db_file, wal_file=self.wal_file, compact_threshold=True)

    def test_rejects_non_int_compact_threshold(self):
        with self.assertRaises(ValueError):
            DbEngine(db_file=self.db_file, wal_file=self.wal_file, compact_threshold="50")

    def test_rejects_empty_db_file_path(self):
        with self.assertRaises(ValueError):
            DbEngine(db_file="", wal_file=self.wal_file)

    def test_rejects_empty_wal_file_path(self):
        with self.assertRaises(ValueError):
            DbEngine(db_file=self.db_file, wal_file="")

    def test_rejects_non_bool_wal_sync_on_write(self):
        with self.assertRaises(ValueError):
            DbEngine(db_file=self.db_file, wal_file=self.wal_file, wal_sync_on_write=1)

    def test_accepts_explicit_valid_overrides(self):
        db = DbEngine(
            db_file=self.db_file,
            wal_file=self.wal_file,
            compact_threshold=10,
            wal_sync_on_write=False,
        )
        self.assertEqual(db.db_file, self.db_file)
        self.assertEqual(db.wal_file, self.wal_file)
        self.assertEqual(db.compact_threshold, 10)
        self.assertEqual(db.wal_sync_on_write, False)


if __name__ == "__main__":
    unittest.main()