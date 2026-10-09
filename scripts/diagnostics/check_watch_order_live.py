"""Opt-in single semantic ordering check; no retries or test discovery."""
import contextlib
import io
import os
import re
from unittest.mock import patch
from fastapi.testclient import TestClient
import backend.api.app as api


def main():
    with patch.dict(os.environ, {'MOVIE_CHAT_MOCK': 'false'}), TestClient(api.app) as client:
        with contextlib.redirect_stdout(io.StringIO()):
            reply = client.post('/movie-chat', json={'message': 'Give me LOTR and The Hobbit movies in release order.', 'country_code': 'SE'})
        data = reply.json()
        print('HTTP:', reply.status_code, 'Status:', data.get('status'))
        print(data.get('assistant_message'))
        print('Tools:', [item['tool'] for item in data.get('tool_results', [])])
        print('Actual completed OpenAI requests:', data.get('openai_requests', 0))
        if reply.status_code != 200 or data.get('status') != 'success':
            return 1
        text = data['assistant_message']
        years = [int(y) for y in re.findall(r'\b(?:19|20)\d{2}\b', text)]
        assert years == [2001, 2002, 2003, 2012, 2013, 2014], years
        assert 'release order' in text.casefold()
        titles = ['fellowship of the ring', 'two towers', 'return of the king',
                  'unexpected journey', 'desolation of smaug', 'battle of the five armies']
        positions = [text.casefold().index(title) for title in titles]
        assert positions == sorted(positions), positions
        assert not data['tool_results'] and data['collection'] is None
        print('PASS: requested six-film scope and ascending release years.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
