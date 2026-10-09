"""Explicit opt-in direct diagnostic: one request, no API/browser/routing or retries."""
import asyncio
import logging
import os
from dotenv import load_dotenv

# Same project loading order as api.py, without importing/loading the engine.
load_dotenv()
os.environ['MOVIE_RESEARCH_DIAGNOSTICS'] = 'true'

from openai import AsyncOpenAI
from chat_service import OPENAI_MODEL
from movie_research import OpenAIWebResearch, MovieResearchRequest, diagnostic_enabled

QUERY = ('Verify whether The Lord of the Rings: The War of the Rohirrim was '
         'released as a theatrical Middle-earth film and give the release year.')


async def main():
    logging.basicConfig(level=logging.WARNING, format='%(message)s')
    if not diagnostic_enabled():
        print('Development diagnostics disabled in production; no request made.')
        return 1
    print('Direct query:', QUERY)
    print('Model:', OPENAI_MODEL)
    async with asyncio.timeout(100), AsyncOpenAI(timeout=30, max_retries=0) as client:
        result = await OpenAIWebResearch(client).research(MovieResearchRequest(message=QUERY,
            policy='Give a concise movie verification answer with sources.'))
    print('Result:', result.status, 'Failure category:', result.failure)
    print('Sources:', len(result.sources), 'Search calls:', result.search_calls,
          'OpenAI request attempts:', result.openai_requests)
    # Source content/answer need not be printed to diagnose the capability.
    return 0 if result.status == 'success' else 1


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
