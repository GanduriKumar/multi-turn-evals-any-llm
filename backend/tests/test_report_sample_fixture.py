import json
from pathlib import Path

from reporter import Reporter


def test_render_with_sample_fixture(tmp_path: Path):
    fixture = Path('backend/tests/fixtures/sample_run_results_min.json')
    data = json.loads(fixture.read_text(encoding='utf-8'))
    rep = Reporter(Path('backend/templates'))
    html = rep.render_html(data)
    out = tmp_path / 'report.html'
    out.write_text(html, encoding='utf-8')
    assert out.exists() and '<!doctype html>' in out.read_text(encoding='utf-8')
