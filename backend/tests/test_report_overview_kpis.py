from pathlib import Path

from reporter import Reporter


def _sample_results():
    # Two conversations, 3 turns total (2 pass, 1 fail)
    return {
        'run_id': 'rid2',
        'dataset_id': 'ds2',
        'model_spec': 'openai:gpt-5',
        'conversations': [
            {
                'conversation_id': 'c1',
                'conversation_title': 'A',
                'summary': {'conversation_pass': True, 'weighted_pass_rate': 1.0, 'final_outcome': {'pass': True, 'reasons': []}},
                'turns': [
                    {'turn_index': 0, 'turn_pass': True, 'metrics': {'exact': {'pass': True}}},
                    {'turn_index': 1, 'turn_pass': True, 'metrics': {'exact': {'pass': True}}},
                ],
            },
            {
                'conversation_id': 'c2',
                'conversation_title': 'B',
                'summary': {'conversation_pass': False, 'weighted_pass_rate': 0.0, 'final_outcome': {'pass': False, 'reasons': ['mismatch']}},
                'turns': [
                    {'turn_index': 0, 'turn_pass': False, 'metrics': {'exact': {'pass': False, 'reason': 'diff'}}},
                ],
            },
        ],
    }


def test_overview_kpis_and_donuts_present():
    rep = Reporter(Path('backend/templates'))
    html = rep.render_html(_sample_results())

    # KPI tiles
    assert 'Conversations Pass' in html
    assert 'Turns Pass' in html
    assert 'Total Conversations' in html
    assert 'Total Turns' in html
    assert 'Failed Turns' in html

    # Donut charts containers
    assert 'class="donut"' in html
    # Percent variables baked into style attribute
    assert '--p:' in html
