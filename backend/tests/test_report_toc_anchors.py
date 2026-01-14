from pathlib import Path

from reporter import Reporter


def _results_two_conversations():
    return {
        'run_id': 'rid4',
        'dataset_id': 'ds4',
        'model_spec': 'openai:gpt-5',
        'conversations': [
            {
                'conversation_id': 'c1',
                'conversation_title': 'First',
                'summary': {'conversation_pass': True},
                'turns': [{'turn_index': 0, 'metrics': {'exact': {'pass': True}}}],
            },
            {
                'conversation_id': 'c2',
                'conversation_title': 'Second',
                'summary': {'conversation_pass': False},
                'turns': [{'turn_index': 0, 'metrics': {'exact': {'pass': False}}}],
            },
        ],
    }


def test_sections_have_ids_and_conversation_anchors_no_toc():
    rep = Reporter(Path('backend/templates'))
    html = rep.render_html(_results_two_conversations())

    # Section ids present
    for sec in ('run-summary', 'overview', 'fail-metrics', 'failure-explanations', 'detailed-report'):
        assert f'id="{sec}"' in html

    # Conversation anchors
    assert 'id="conv-c1"' in html
    assert 'id="conv-c2"' in html

    # Back to top link present
    assert 'href="#top"' in html

    # TOC removed
    assert 'class="toc"' not in html and '>Contents<' not in html
