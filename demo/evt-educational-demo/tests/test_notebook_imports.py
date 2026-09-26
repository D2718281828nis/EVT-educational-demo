"""Regression tests for VS Code/Jupyter import bootstrapping."""

import json
from pathlib import Path
import subprocess
import sys


def test_every_notebook_first_code_cell_imports_evt_demo():
    demo_root = Path(__file__).resolve().parents[1]
    repository_root = demo_root.parents[1]
    notebooks = sorted((demo_root / "notebooks").glob("*.ipynb"))
    assert len(notebooks) == 6
    for notebook in notebooks:
        document = json.loads(notebook.read_text(encoding="utf-8"))
        first_code = next(
            cell for cell in document["cells"] if cell["cell_type"] == "code"
        )
        source = "".join(first_code["source"])
        assert "sys.path.insert" in source
        lines = source.splitlines()
        last_import = max(
            index for index, line in enumerate(lines) if "from evt_demo" in line
        )
        import_only_source = "\n".join(lines[: last_import + 1])
        subprocess.run(
            [sys.executable, "-c", import_only_source],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            [sys.executable, "-c", import_only_source],
            cwd=notebook.parent,
            check=True,
            capture_output=True,
            text=True,
        )
