"""Opt-in two informational messages, once; no retries or recommendations."""
import contextlib
import io
import os
from unittest.mock import patch
from fastapi.testclient import TestClient
import api


def main():
    count = 0
    cases = ['Give me the X-Men collection in release order.',
        'I want all the movies of X-Men universe including Deadpool for example in release order.']
    with patch.dict(os.environ, {'MOVIE_CHAT_MOCK': 'false'}), TestClient(api.app) as client:
        for index, message in enumerate(cases):
            with contextlib.redirect_stdout(io.StringIO()):
                reply = client.post('/movie-chat', json={'message': message, 'country_code': 'SE'})
            data = reply.json()
            count += data.get('openai_requests', 0)
            print(message)
            print('HTTP:', reply.status_code, 'Status:', data.get('status'), 'Requests:', data.get('openai_requests', 0))
            print(data.get('assistant_message'))
            print('Tools:', [item['tool'] for item in data.get('tool_results', [])])
            print('Total completed OpenAI requests:', count)
            if reply.status_code != 200 or data.get('status') != 'success':
                return 1
            if index == 0:
                assert data['collection']['source'] == 'TMDB'
                assert [item['tool'] for item in data['tool_results']] == ['search_movie_collection', 'get_movie_collection']
            else:
                assert data['collection'] is None and not data['tool_results']
                text = data['assistant_message'].casefold()
                assert 'deadpool' in text and 'wolverine' in text and 'new mutants' in text
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
