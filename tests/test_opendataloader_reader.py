"""OpenDataLoader adapter tests."""

from pathlib import Path
from types import SimpleNamespace

from src.pdf_pipeline.opendataloader_reader import _run_opendataloader


def test_run_opendataloader_uses_ascii_temp_pdf_for_cli(tmp_path, monkeypatch) -> None:
    pdf_name = (
        "\u5c3f\u8def\u611f\u67d3\u8bca\u65ad\u4e0e\u6cbb\u7597"
        "\u4e2d\u56fd\u4e13\u5bb6\u5171\u8bc6\u2014\u590d\u6742\u6027"
        "\u5c3f\u8def\u611f\u67d3.pdf"
    )
    pdf_path = tmp_path / pdf_name
    pdf_path.write_bytes(b"%PDF-1.4\n")
    output_dir = tmp_path / "processed" / pdf_path.stem
    captured_args = {}

    def fake_run(args, **kwargs):
        captured_args["args"] = args
        cli_pdf = args[1]
        cli_output_dir = Path(args[3])
        assert cli_pdf.endswith("input.pdf")
        assert "\u5c3f\u8def\u611f\u67d3" not in cli_pdf
        result_json = cli_output_dir / "result.json"
        result_json.parent.mkdir(parents=True, exist_ok=True)
        result_json.write_text(
            '{"elements": [{"type": "paragraph", "text": "body", "page_number": 1}]}',
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("src.pdf_pipeline.opendataloader_reader.subprocess.run", fake_run)

    json_path = _run_opendataloader(Path(f"{pdf_path}\n"), output_dir, {"command": "opendataloader-pdf"})

    assert captured_args["args"][2] == "-o"
    assert json_path == output_dir / "result.json"
    assert json_path.exists()
