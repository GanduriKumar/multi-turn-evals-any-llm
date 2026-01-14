from pathlib import Path

from reporter import Reporter


def _results_one_conversation_two_turns():
    return {
        'run_id': 'rid5',
        'dataset_id': 'ds5',
        'model_spec': 'openai:gpt-5',
        'conversations': [
            {
                'conversation_id': 'c1',
                'conversation_title': 'Turns Layout',
                'summary': {'conversation_pass': True},
                'turns': [
                    {
                        'turn_index': 0,
                        'turn_pass': True,
                        'user_prompt_snippet': 'Hello',
                        'assistant_output_snippet': 'Hi!'
                    },
                    {
                        'turn_index': 1,
                        'turn_pass': False,
                        'user_prompt_snippet': 'What is 2+2?',
                        'assistant_output_snippet': '5'
                    }
                ],
            },
        ],
    }


def test_turn_cards_and_two_column_grid_present():
    rep = Reporter(Path('backend/templates'))
    html = rep.render_html(_results_one_conversation_two_turns())

    assert 'class="turns"' in html
    assert 'class="turn-grid"' in html
    assert 'User Prompt' in html and 'Assistant Output' in html
    # Badge reflects pass/fail
    assert 'badge-pass' in html or 'badge-fail' in html
