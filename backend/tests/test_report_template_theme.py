from pathlib import Path

from reporter import Reporter


def _minimal_results():
    return {
        'run_id': 'rid',
        'dataset_id': 'ds',
        'model_spec': 'openai:gpt-5',
        'conversations': [
            {
                'conversation_id': 'c1',
                'conversation_title': 'Sample Conversation',
                'summary': {
                    'conversation_pass': True,
                    'weighted_pass_rate': 1.0,
                    'final_outcome': {'pass': True, 'reasons': []},
                },
                'turns': [
                    {
                        'turn_index': 0,
                        'user_prompt_snippet': 'Hi',
                        'assistant_output_snippet': 'Hello',
                        'metrics': {'exact': {'pass': True}},
                    }
                ],
            }
        ],
    }


def test_template_contains_brand_css_and_sticky_headers():
    rep = Reporter(Path('backend/templates'))
    html = rep.render_html(_minimal_results())

    # Brand CSS variables present
    assert '--g-blue: #4285F4' in html
    assert '--g-red: #EA4335' in html
    assert '--g-yellow: #FBBC05' in html
    assert '--g-green: #34A853' in html

    # Sticky table headers and zebra striping
    assert 'position: sticky' in html
    assert 'tbody tr:nth-child(odd)' in html


def test_template_is_self_contained_no_external_assets():
    rep = Reporter(Path('backend/templates'))
    html = rep.render_html(_minimal_results())

    # No external links or scripts (allow anchors only)
    assert '<script' not in html
    assert 'href="http' not in html and 'href="https' not in html
    assert 'src="http' not in html and 'src="https' not in html
