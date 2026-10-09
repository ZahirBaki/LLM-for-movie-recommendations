"""Offline diagnostic safety and sanitized real-SDK response-shape regressions."""
import json
import os
import unittest
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace as NS
import httpx
from openai import BadRequestError
from openai.types.responses import Response, ResponseFunctionWebSearch, ResponseOutputMessage, ResponseOutputText
from openai.types.responses.response_output_text import AnnotationURLCitation
from movie_research import (local_diagnostic, sanitized_diagnostic_text,
    OpenAIWebResearch, MovieResearchRequest, parse_research_response)


def observed_response_fixture():
    # Geometry/types/counts observed in the direct successful diagnostic. Actual
    # returned text, links, IDs and query are replaced with neutral test data.
    spans = [(264, 382), (663, 781), (905, 1023), (1097, 1227)]
    annotations = [AnnotationURLCitation(type='url_citation', start_index=start, end_index=end,
        title='Neutral source', url=f'https://example.org/source-{0 if i < 3 else 1}')
        for i, (start, end) in enumerate(spans)]
    part = ResponseOutputText(type='output_text', text='x' * 1229, annotations=annotations)
    message = ResponseOutputMessage(type='message', id='msg_fixture', status='completed', role='assistant', content=[part])
    search = ResponseFunctionWebSearch(type='web_search_call', id='ws_fixture', status='completed', action={'type': 'search', 'query': 'neutral film query'})
    return Response.model_construct(status='completed', output=[search, message], incomplete_details=None)


class ResearchDiagnosticTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        patch.dict(os.environ, {'MOVIE_RESEARCH_DIAGNOSTICS': 'true', 'MOVIE_RESEARCH_ENABLED': 'true',
            'ENVIRONMENT': 'development', 'APP_ENV': 'development', 'NODE_ENV': 'development'}).start()
        self.addCleanup(patch.stopall)

    async def test_observed_sdk_structure_parses_completed_search_and_four_citations(self):
        result = parse_research_response(observed_response_fixture())
        self.assertEqual(result.status, 'success')
        self.assertEqual(result.search_calls, 1)
        self.assertEqual(len(result.sources), 2)
        self.assertEqual(len(result.citations), 4)

    async def test_metadata_is_logged_without_web_text_or_prompt(self):
        response = observed_response_fixture()
        with self.assertLogs('movie_research', level='WARNING') as logs:
            local_diagnostic('response_received', response=response)
        data = json.loads(logs.output[0].split('[MovieResearch] ', 1)[1])
        self.assertEqual(data['response_status'], 'completed')
        self.assertEqual(data['output_types'], ['web_search_call', 'message'])
        self.assertEqual(data['search_count'], 1)
        self.assertEqual(data['citation_count'], 4)
        self.assertTrue(data['text_returned'])
        self.assertNotIn(response.output[1].content[0].text, logs.output[0])
        self.assertNotIn('neutral film query', logs.output[0])

    async def test_http_error_fields_are_safe_and_never_enter_result(self):
        error = BadRequestError('unused SDK wrapper', response=httpx.Response(400,
            headers={'x-request-id': 'req_fixture'}, request=httpx.Request('POST', 'https://api.openai.com/v1/responses')),
            body={'error': {'type': 'invalid_request_error', 'code': 'invalid_value',
                'message': 'Invalid tool selection. OPENAI_API_KEY=offline-secret Cookie: session-secret'}})
        client = NS(responses=NS(create=AsyncMock(side_effect=error)))
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'offline-secret'}), self.assertLogs('movie_research', level='WARNING') as logs:
            result = await OpenAIWebResearch(client).research(MovieResearchRequest(message='Verify a movie.', policy='PRIVATE SYSTEM PROMPT'))
        data = json.loads(next(line for line in logs.output if '[MovieResearch]' in line).split('[MovieResearch] ', 1)[1])
        self.assertEqual(data['exception'], 'BadRequestError')
        self.assertEqual(data['status_code'], 400)
        self.assertEqual(data['error_type'], 'invalid_request_error')
        self.assertEqual(data['error_code'], 'invalid_value')
        self.assertEqual(data['request_id'], 'req_fixture')
        for secret in ['offline-secret', 'session-secret', 'PRIVATE SYSTEM PROMPT', 'Authorization']:
            self.assertNotIn(secret, ' '.join(logs.output))
            self.assertNotIn(secret, result.model_dump_json())
        self.assertEqual(result.status, 'unavailable')
        client.responses.create.assert_awaited_once()

    async def test_detailed_diagnostics_disabled_by_default_and_in_production(self):
        for settings in [{'MOVIE_RESEARCH_DIAGNOSTICS': 'false'}, {'ENVIRONMENT': 'production'},
                         {'APP_ENV': 'production'}, {'NODE_ENV': 'production'}]:
            with patch.dict(os.environ, settings), patch('movie_research.logger.warning') as log:
                local_diagnostic('request_failed', error=RuntimeError('private'))
            log.assert_not_called()

    async def test_json_headers_and_credential_urls_are_redacted(self):
        value = '{"Authorization": "Bearer abc123", "Cookie": "session=private", "api_key": "sk-test-value"} https://user:privatepass@example.org'
        cleaned = sanitized_diagnostic_text(value)
        for secret in ['abc123', 'session=private', 'sk-test-value', 'privatepass']:
            self.assertNotIn(secret, cleaned)
        self.assertIn('[REDACTED]', cleaned)

    async def test_incomplete_no_citations_and_malformed_outputs_stay_safe(self):
        fixtures = [observed_response_fixture(), observed_response_fixture(), observed_response_fixture()]
        fixtures[0].status = 'incomplete'
        fixtures[0].incomplete_details = NS(reason='max_output_tokens')
        fixtures[1].output[1].content[0].annotations = []
        fixtures[2].output = [NS(type='function_call', arguments='private payload')]
        for response in fixtures:
            client = NS(responses=NS(create=AsyncMock(return_value=response)))
            with self.assertLogs('movie_research', level='WARNING') as logs:
                result = await OpenAIWebResearch(client).research(MovieResearchRequest(message='Verify a movie.', policy='Private policy'))
            self.assertEqual(result.status, 'unavailable')
            self.assertEqual(result.sources, [])
            self.assertNotIn('private payload', ' '.join(logs.output))
            self.assertNotIn('Private policy', ' '.join(logs.output))
            self.assertEqual(result.openai_requests, 1)
