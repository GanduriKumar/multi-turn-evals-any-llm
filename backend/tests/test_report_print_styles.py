from pathlib import Path

from reporter import Reporter


def _results_small():
    return {
        'run_id': 'rid7',
        'dataset_id': 'ds7',
        'model_spec': 'openai:gpt-5',
        'conversations': [
            {'conversation_id': 'c1', 'summary': {'conversation_pass': True}, 'turns': [{'turn_index': 0, 'metrics': {'exact': {'pass': True}}}]}
        ],
    }


def test_print_media_rules_present():
    rep = Reporter(Path('backend/templates'))
    html = rep.render_html(_results_small())
    assert '@media print' in html
    assert 'break-before: page' in html or 'page-break-before: always' in html
    assert '.toc' in html and 'display: none' in html
