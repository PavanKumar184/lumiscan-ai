from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent
    venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    app_file = project_root / "app.py"

    if not venv_python.exists():
        raise FileNotFoundError(
            f"Virtual environment Python not found at: {venv_python}"
        )

    if not app_file.exists():
        raise FileNotFoundError(f"Flask app file not found at: {app_file}")

    print(f"Starting app using: {venv_python}")
    print(f"App file: {app_file}")
    print("Open http://127.0.0.1:5000 in your browser after startup.\n")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        [str(venv_python), str(app_file)],
        cwd=str(project_root),
        env=env,
        check=False,
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
