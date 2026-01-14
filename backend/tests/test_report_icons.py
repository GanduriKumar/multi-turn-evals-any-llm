from pathlib import Path

from reporter import Reporter


def _results_one_turn():
    return {
        'run_id': 'rid6',
        'dataset_id': 'ds6',
        'model_spec': 'openai:gpt-5',
        'conversations': [
            {
                'conversation_id': 'c1',
                'summary': {'conversation_pass': True},
                'turns': [
                    {'turn_index': 0, 'turn_pass': True, 'user_prompt_snippet': 'Hi', 'assistant_output_snippet': 'Hello'}
                ],
            }
        ],
    }


def test_inline_svg_symbols_and_usage_present():
    rep = Reporter(Path('backend/templates'))
    html = rep.render_html(_results_one_turn())

    # Symbols defined
    assert 'id="ic-check"' in html
    assert 'id="ic-x"' in html

    # Used in badges
    assert '<use href="#ic-check"' in html or '<use href="#ic-x"' in html
