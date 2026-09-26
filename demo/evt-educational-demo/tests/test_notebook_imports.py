"""Structural and execution-contract checks that do not contact network services."""

import json
from pathlib import Path

import graph_evt_agent


PUBLIC_PIPELINE_NAMES = {
    "EVTConfig",
    "GraphConfig",
    "GraphEVTPipeline",
    "InputConfig",
}


def test_notebooks_share_dataset_and_use_supported_library_api():
    demo_root = Path(__file__).resolve().parents[1]
    notebooks = sorted((demo_root / "notebooks").glob("*.ipynb"))

    assert [path.name[:2] for path in notebooks] == ["01", "02", "03", "04", "05", "06", "07"]
    for notebook in notebooks:
        document = json.loads(notebook.read_text(encoding="utf-8"))
        assert all(not cell.get("outputs") for cell in document["cells"])
        source = "\n".join("".join(cell["source"]) for cell in document["cells"])
        expected_data = "kuramoto_synchronized_series.csv" if notebook.name.startswith("07") else "fractal_extreme_series.csv"
        assert expected_data in source
        assert "SEED = 42" in source
        assert "sys.path.insert" not in source

    for notebook in notebooks[2:]:
        source = notebook.read_text(encoding="utf-8")
        assert "from graph_evt_agent import" in source
        assert "GraphEVTPipeline" in source
        assert "ResearchAgent" not in source
        assert "ResearchTask" not in source
        assert "LLMConfig" not in source


def test_llm_notebook_uses_graph_evt_agent_team_without_embedding_a_key():
    notebook = Path(__file__).resolve().parents[1] / "notebooks/04_llm_hypothesis_agent.ipynb"
    source = notebook.read_text(encoding="utf-8")
    assert "EVTAgentTeam" in source
    assert "MistralClient" in source
    assert 'os.getenv(\\"MISTRAL_API_KEY\\")' in source
    assert "Mistral API key (input hidden)" in source
    assert "MistralClient(api_key=api_key)" in source


def test_visualization_notebook_bootstraps_local_src_before_import():
    notebook = Path(__file__).resolve().parents[1] / "notebooks/07_detection_and_source_visualization.ipynb"
    document = json.loads(notebook.read_text(encoding="utf-8"))
    source = "".join(document["cells"][1]["source"])

    assert 'sys.path.append(src_path)' in source
    assert source.index('sys.path.append(src_path)') < source.index(
        "from evt_demo.visualization import plot_evt_source_result"
    )


def test_documented_pipeline_symbols_are_exported():
    missing = PUBLIC_PIPELINE_NAMES - set(dir(graph_evt_agent))
    assert not missing


def test_research_dataset_schema_and_size():
    data_path = Path(__file__).resolve().parents[1] / "data/fractal_extreme_series.csv"
    lines = data_path.read_text(encoding="utf-8").splitlines()

    assert lines[0] == "time,value"
    assert len(lines) == 4097
    assert not (data_path.parent / "synthetic_data.csv").exists()
