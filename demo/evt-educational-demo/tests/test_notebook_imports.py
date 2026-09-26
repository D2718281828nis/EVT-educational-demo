"""Structural checks that do not spend tokens or contact an LLM provider."""

import json
from pathlib import Path


def test_renewed_notebooks_share_dataset_and_external_agent_library():
    demo_root = Path(__file__).resolve().parents[1]
    notebooks = sorted((demo_root / "notebooks").glob("*.ipynb"))

    assert [path.name[:2] for path in notebooks] == ["01", "02", "03", "04", "05", "06"]
    for notebook in notebooks:
        document = json.loads(notebook.read_text(encoding="utf-8"))
        assert all(not cell.get("outputs") for cell in document["cells"])
        source = "\n".join(
            "".join(cell["source"]) for cell in document["cells"]
        )
        assert "fractal_extreme_series.csv" in source
        assert "SEED = 42" in source
        assert "sys.path.insert" not in source

    agent_notebooks = notebooks[2:]
    for notebook in agent_notebooks:
        source = notebook.read_text(encoding="utf-8")
        assert "from graph_evt_agent import" in source
        assert "ResearchTask" in source


def test_research_dataset_schema_and_size():
    data_path = Path(__file__).resolve().parents[1] / "data/fractal_extreme_series.csv"
    lines = data_path.read_text(encoding="utf-8").splitlines()

    assert lines[0] == "time,value"
    assert len(lines) == 4097
    assert not (data_path.parent / "synthetic_data.csv").exists()
