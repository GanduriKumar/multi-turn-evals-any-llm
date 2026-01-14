from pathlib import Path

from reporter import Reporter


def test_render_with_minimal_results():
    rep = Reporter(Path('backend/templates'))
    # Minimal viable structure
    html = rep.render_html({'run_id': 'r', 'conversations': []})
    assert '<html>' in html and 'Run r' in html
    # Ensure sections render
    assert 'Run Summary' in html
    assert 'Overview' in html
    assert 'Detailed Report' in html


def test_render_with_missing_fields_and_metrics():
    rep = Reporter(Path('backend/templates'))
    run = {
        'run_id': 'r2',
        # dataset_id/model_spec missing intentionally
        'conversations': [
            {
                'conversation_id': 'c1'
                # summary and turns missing intentionally
            }
        ]
    }
    html = rep.render_html(run)
    # Should not crash; should include conversation anchor and title fallback
    assert 'id="conv-c1"' in html or 'Conversation 1' in html
