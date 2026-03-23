"""Launch the Streamlit folding demo (install: pip install 'protein-folder[demo]')."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    ui = Path(__file__).resolve().with_name("demo_streamlit_ui.py")
    if not ui.is_file():
        print("demo_streamlit_ui.py missing next to demo_app.py", file=sys.stderr)
        raise SystemExit(1)
    cmd = [sys.executable, "-m", "streamlit", "run", str(ui), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
