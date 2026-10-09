"""Test-only hosted search response fixture. Never used by application code."""
from types import SimpleNamespace as NS


def research_response(text):
    # Citation text appears in the answer, so no uncited model prose is presented
    # as a successful hosted-search response in controller tests.
    text += ' [Source]'
    annotation = NS(type='url_citation', title='Fixture movie source', url='https://example.org/movies',
        start_index=len(text) - 8, end_index=len(text))
    return NS(status='completed', output=[NS(type='web_search_call', status='completed'),
        NS(type='message', content=[NS(type='output_text', text=text, annotations=[annotation])])])
