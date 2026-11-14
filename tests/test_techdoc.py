"""WS1 (saha.tools.techdoc): the technical document is assembled from docs/tech/."""

from saha.tools import techdoc


def write_tech_dir(tmp_path):
    (tmp_path / "saha_main.md").write_text(
        "# SAHA\n\nMain text.\n\n## Appendices\n\nEach package's section, as its owner wrote it.\n",
        encoding="utf-8")
    (tmp_path / "ws1_pipeline.md").write_text("# WS1: pipeline\n\n## Approach\n\nLong text.\n",
                                              encoding="utf-8")
    (tmp_path / "ws3_visible.md").write_text("# WS3: visible\n\nMore text.\n", encoding="utf-8")
    (tmp_path / "appendix_summaries.md").write_text(
        "## Appendices\n\nIn brief.\n\n### Appendix A: WS1\n\nShort.\n", encoding="utf-8")
    return tmp_path


def test_the_default_document_has_the_short_appendices_only(tmp_path):
    text = techdoc.assemble(write_tech_dir(tmp_path))
    assert text.count("## Appendices") == 1
    assert "In brief." in text and "Short." in text
    assert "Long text." not in text and "as its owner wrote it" not in text
    assert text.startswith("# SAHA\n\nMain text.\n")


def test_the_full_document_appends_each_section_in_ws_order(tmp_path):
    text = techdoc.assemble(write_tech_dir(tmp_path), full=True)
    assert "In brief." not in text
    a, c = text.index("## Appendix A: WS1: pipeline"), text.index("## Appendix C: WS3: visible")
    assert a < c
    assert "### Approach" in text                     # headings move one level down
