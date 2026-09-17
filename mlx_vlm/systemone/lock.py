"""A single-instance lock for the System One server.

The model is tens of gigabytes, so a second server is not merely redundant: two
of them race for memory and the kernel kills one mid-request. The lock makes
that failure a clear message at startup instead of an OOM later.

Stale locks are reclaimed. A lock file left by a process that has since died
would otherwise keep the port unusable until someone deleted it by hand.
"""

from __future__ import annotations

import errno
import json
import os
import tempfile
from pathlib import Path
from typing import Optional


class ServerAlreadyRunning(RuntimeError):
    def __init__(self, pid: int, port: Optional[int], path: Path):
        self.pid = pid
        self.port = port
        where = f" on port {port}" if port else ""
        super().__init__(
            f"A System One server is already running{where} (pid {pid}).\n"
            f"Stop it with:  kill {pid}\n"
            f"Lock file:     {path}"
        )


def _process_alive(pid: int) -> bool:
    """True if a process with this pid exists and we may signal it."""
    try:
        os.kill(pid, 0)
    except OSError as exc:
        # EPERM means it exists but belongs to someone else, which still counts.
        return exc.errno == errno.EPERM
    return True


def default_lock_path() -> Path:
    return Path(tempfile.gettempdir()) / "mlx_vlm_systemone.lock"


class SingleInstanceLock:
    """Refuse to start when another live server holds the lock.

    Used as a context manager so the lock is released on any exit path,
    including an exception during model load.
    """

    def __init__(self, path: Optional[Path] = None, port: Optional[int] = None):
        self.path = Path(path or default_lock_path())
        self.port = port
        self._acquired = False

    def _read(self) -> Optional[dict]:
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return None

    def acquire(self) -> "SingleInstanceLock":
        if self.path.exists():
            existing = self._read()
            pid = int((existing or {}).get("pid", -1))
            # A live holder blocks us even if it is this process: acquiring
            # twice means a caller lost track of a server it already started.
            if pid > 0 and _process_alive(pid):
                raise ServerAlreadyRunning(pid, (existing or {}).get("port"), self.path)
            # Dead holder, or a file we cannot parse, tells us nothing worth
            # honouring — either way it must not wedge the port forever.
            self.path.unlink(missing_ok=True)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        # O_EXCL so two servers starting together cannot both believe they won.
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            existing = self._read() or {}
            raise ServerAlreadyRunning(
                int(existing.get("pid", -1)), existing.get("port"), self.path
            ) from None
        with os.fdopen(fd, "w") as handle:
            json.dump({"pid": os.getpid(), "port": self.port}, handle)
        self._acquired = True
        return self

    def release(self) -> None:
        if not self._acquired:
            return
        # Only remove our own lock: a stale-reclaim race could mean the file now
        # belongs to a different server.
        current = self._read()
        if current and int(current.get("pid", -1)) == os.getpid():
            self.path.unlink(missing_ok=True)
        self._acquired = False

    def __enter__(self) -> "SingleInstanceLock":
        return self.acquire()

    def __exit__(self, *exc_info) -> None:
        self.release()
