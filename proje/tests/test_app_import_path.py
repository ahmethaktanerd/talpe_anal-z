import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_app_entrypoint_when_vscode_puts_app_directory_first():
    code = f"""
import runpy
import sys

root = {str(ROOT)!r}
app_dir = root + "/app"
sys.path = [app_dir, root] + [
    entry for entry in sys.path if entry not in {{app_dir, root}}
]
runpy.run_path(root + "/app/app.py", run_name="__main__")
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT / "app",
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
