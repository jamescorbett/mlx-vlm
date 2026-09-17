"""Single-instance lock for the System One server.

The model is tens of gigabytes. Two servers race for memory and the kernel
kills one mid-request, so the lock turns that into a message at startup.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from mlx_vlm.systemone.lock import (
    ServerAlreadyRunning,
    SingleInstanceLock,
    _process_alive,
    default_lock_path,
)


class TestSingleInstanceLock(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "systemone.lock"

    def tearDown(self):
        self.dir.cleanup()

    def test_acquiring_records_our_pid_and_port(self):
        with SingleInstanceLock(self.path, port=8100):
            recorded = json.loads(self.path.read_text())
            self.assertEqual(recorded["pid"], os.getpid())
            self.assertEqual(recorded["port"], 8100)

    def test_the_lock_is_released_on_exit(self):
        with SingleInstanceLock(self.path):
            self.assertTrue(self.path.exists())
        self.assertFalse(self.path.exists())

    def test_the_lock_is_released_even_when_startup_fails(self):
        # Model loading can fail; the lock must not outlive the attempt.
        with self.assertRaises(RuntimeError):
            with SingleInstanceLock(self.path):
                raise RuntimeError("model load failed")
        self.assertFalse(self.path.exists())

    def test_a_live_holder_blocks_a_second_server(self):
        held = SingleInstanceLock(self.path, port=8100).acquire()
        try:
            with self.assertRaises(ServerAlreadyRunning) as caught:
                SingleInstanceLock(self.path, port=8100).acquire()
            message = str(caught.exception)
            self.assertIn(str(os.getpid()), message)
            self.assertIn("8100", message)
            self.assertIn("kill", message)
        finally:
            held.release()

    def test_a_stale_lock_is_reclaimed(self):
        # A lock left by a dead process must not wedge the port forever. pid 1
        # is alive, so use a pid that cannot be: one past the max.
        dead = 2**31 - 1
        self.path.write_text(json.dumps({"pid": dead, "port": 8100}))
        self.assertFalse(_process_alive(dead))
        with SingleInstanceLock(self.path, port=8100):
            self.assertEqual(json.loads(self.path.read_text())["pid"], os.getpid())

    def test_a_corrupt_lock_is_reclaimed(self):
        self.path.write_text("not json at all")
        with SingleInstanceLock(self.path, port=8100):
            self.assertEqual(json.loads(self.path.read_text())["pid"], os.getpid())

    def test_releasing_a_lock_we_no_longer_own_leaves_it_alone(self):
        # After a stale-reclaim race the file may belong to a different server;
        # releasing ours must not delete theirs.
        lock = SingleInstanceLock(self.path).acquire()
        self.path.write_text(json.dumps({"pid": 999_999, "port": 8100}))
        lock.release()
        self.assertTrue(self.path.exists())

    def test_release_without_acquire_is_harmless(self):
        SingleInstanceLock(self.path).release()
        self.assertFalse(self.path.exists())

    def test_the_default_path_is_shared_between_invocations(self):
        self.assertEqual(default_lock_path(), default_lock_path())
        self.assertTrue(str(default_lock_path()).endswith(".lock"))

    def test_a_live_process_is_detected(self):
        self.assertTrue(_process_alive(os.getpid()))


if __name__ == "__main__":
    unittest.main()
