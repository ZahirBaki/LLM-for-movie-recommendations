"""One opt-in paid research check. No retries and no test-discovery execution."""
import contextlib
import io
import os
from unittest.mock import patch
from fastapi.testclient import TestClient
import backend.api.app as api


def main():
    with patch.dict(os.environ, {'MOVIE_CHAT_MOCK': 'false', 'MOVIE_RESEARCH_ENABLED': 'true'}), TestClient(api.app) as client:
        with patch.object(api.engine, 'recommend', wraps=api.engine.recommend) as ranking:
            with contextlib.redirect_stdout(io.StringIO()):
                reply = client.post('/movie-chat', json={'message': 'Give me all LOTR/Middle-earth universe movies in release order.', 'country_code': 'SE'})
            data = reply.json()
            print('HTTP:', reply.status_code, 'Status:', data.get('status'))
            print(data.get('assistant_message'))
            print('Sources:', data.get('sources', []))
            print('Actual OpenAI request attempts:', data.get('openai_requests', 0))
            ranking.assert_not_called()
            if reply.status_code != 200 or data.get('status') != 'success':
                print('Research unavailable or failed. No retries or memory fallback.')
                return 1
            assert 'war of the rohirrim' in data['assistant_message'].casefold()
            assert data['sources'] and data['citations'] and not data['recommendations']
            print('PASS: cited research includes The War of the Rohirrim; no recommendation bypass.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
