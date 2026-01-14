from pathlib import Path

from reporter import Reporter


def _results_with_failures():
    return {
        'run_id': 'rid3',
        'dataset_id': 'ds3',
        'model_spec': 'openai:gpt-5',
        'conversations': [
            {
                'conversation_id': 'c1',
                'conversation_title': 'C1',
                'summary': {'conversation_pass': False, 'weighted_pass_rate': 0.0, 'final_outcome': {'pass': False, 'reasons': ['bad'] }},
                'turns': [
                    {'turn_index': 0, 'turn_pass': False, 'metrics': {'exact': {'pass': False, 'reason': 'diff'}}},
                    {'turn_index': 1, 'turn_pass': False, 'metrics': {'semantic': {'pass': False, 'score_max': 0.21}}},
                ],
            },
            {
                'conversation_id': 'c2',
                'conversation_title': 'C2',
                'summary': {'conversation_pass': False, 'weighted_pass_rate': 0.0, 'final_outcome': {'pass': False, 'reasons': ['bad'] }},
                'turns': [
                    {'turn_index': 0, 'turn_pass': False, 'metrics': {'adherence': {'pass': False}}},
                    {'turn_index': 1, 'turn_pass': False, 'metrics': {'consistency': {'pass': False, 'reason': 'contradiction'}}},
                    {'turn_index': 2, 'turn_pass': False, 'metrics': {'hallucination': {'pass': False, 'reason': 'fabricated'}}},
                ],
            },
        ],
    }


def test_failures_by_metric_cards_and_counts():
    rep = Reporter(Path('backend/templates'))
    html = rep.render_html(_results_with_failures())

    # Section header exists
    assert 'Failures by Metric' in html

    # Each metric has a card and count attribute
    for metric in ('exact', 'semantic', 'consistency', 'adherence', 'hallucination'):
        assert f'data-metric="{metric}"' in html
        assert 'data-failed-count="' in html

    # Links/anchors present
    assert '#metric-exact' in html and 'id="metric-exact"' in html
    assert '#metric-semantic' in html and 'id="metric-semantic"' in html
