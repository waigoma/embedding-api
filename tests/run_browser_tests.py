"""Start and clean up a task-owned loopback fixture for repeatable browser tests."""
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:19975/health"


def reachable():
    try:
        with urllib.request.urlopen(URL, timeout=1) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def main():
    if reachable():
        raise RuntimeError("Fixture port is in use; stop that local process first.")
    environment = {**os.environ, "UI_ARTIFACT_DIR": os.environ.get("UI_ARTIFACT_DIR", str(ROOT / "artifacts"))}
    with tempfile.TemporaryFile(mode="w+") as log:
        fixture = subprocess.Popen([sys.executable, str(ROOT / "tests/serve_ui_fixture.py")], cwd=ROOT, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 20
            while not reachable():
                if fixture.poll() is not None or time.monotonic() >= deadline:
                    log.seek(0)
                    raise RuntimeError(f"Fixture failed to start:\n{log.read()}")
                time.sleep(0.1)
            subprocess.run(["node", "tests/test_admin_browser.cjs"], cwd=ROOT, env=environment, check=True, timeout=120)
        finally:
            fixture.terminate()
            try:
                fixture.wait(timeout=5)
            except subprocess.TimeoutExpired:
                fixture.kill()
                fixture.wait(timeout=5)


if __name__ == "__main__":
    main()
