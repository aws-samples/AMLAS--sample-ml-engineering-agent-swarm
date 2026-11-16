"""Pinned selection after a metadata page; no real transport or native job."""

from test_huggingface_ingestion import PAYLOAD, REVISION, guarded, wire  # noqa: F401

from amlas.tools.research import public_fetch


def test_pinned_acquisition_selects_listed_file_beyond_initial_metadata_page(request, monkeypatch, tmp_path):
    module, _calls = request.getfixturevalue("guarded")
    info = wire()
    info["siblings"] = [{**info["siblings"][0], "rfilename": f"data/{i:04d}.jsonl"} for i in range(700)]
    monkeypatch.setattr(public_fetch, "fetch_public_json", lambda *args, **kwargs: info)
    result = module.acquire_huggingface_dataset_local(
        "example/coding", REVISION, "data/0699.jsonl", str(tmp_path)
    )
    assert result["success"] is True, "pinned acquisition only searched the first metadata page"
    from pathlib import Path

    assert Path(result["local_path"]).read_bytes() == PAYLOAD
