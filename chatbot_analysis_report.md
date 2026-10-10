hugya # XICTEK Systems Chatbot Analysis Report

**Date:** 2026-10-10  
**Project:** `Ai-chatbot`  
**Scope:** Existing Python CLI and FastAPI chatbot code, plus the public XICTEK Systems website.

## Executive summary

The local project contained multiple chatbot entry points with inconsistent prompts, a knowledge base unrelated to XICTEK Systems, unfiltered vector retrieval, unsafe error responses, and import-time interactive loops. The public XICTEK assistant also accepted a test question but did not return an answer during a 40-second observation.

The local chatbot now uses business information checked against the public XICTEK homepage and services page, filters low-relevance retrieval results, returns sanitized service errors, limits stored chat history, and protects knowledge-management routes with an API key. Ten local tests pass. No live Gemini API calls were made, so real-model answer quality and embedding behavior still need deployment-level evaluation.

## Website and business context verified

Reviewed:

- [XICTEK Systems homepage](https://www.xicteksystems.com/)
- [XICTEK Systems services](https://www.xicteksystems.com/services)

The public pages describe XICTEK Systems as a product-focused software company. Listed services include custom software, web and mobile applications, SaaS, AI and automation, cloud solutions, ERP/business automation, DevOps/cloud infrastructure, API integrations, and UI/UX design. The services page mentions React, Next.js, TypeScript, Flutter, React Native, Azure, AWS, LLM integrations, chatbots, and custom ML.

The homepage describes HisabDo as an expense-management and digital-ledger product for individuals and small businesses. Listed features include offline-first records, digital khata and customer records, PDF reports, voice entry, and backup/restore. The listed contact address is `xicteksystems@gmail.com`.

These facts are represented in the project as sourced context; they should be reviewed when the website changes.

## Issues identified and root causes

| Issue | Root cause | Evidence |
|---|---|---|
| Business answers could be unrelated or fabricated | The local RAG knowledge base described unsupported refund, business-hours, and quotation policies rather than XICTEK Systems or HisabDo. | Original `AI_service/chat_api.py` seed records did not match the supplied website. |
| Irrelevant facts could be returned for any question | Retrieval always returned the three highest cosine scores, with no minimum relevance cutoff. | Original `search_knowledge()` returned `results[:3]` unconditionally. |
| The current user question was duplicated in the prompt history | The endpoint appended the current message before formatting history, then added it again as the current question. | Original `/chat` flow. |
| API errors looked like successful responses and exposed provider details | The endpoint returned a normal JSON error object and included raw exception text after retrying all exception types. | Original `/chat` exception branch. |
| Knowledge could be read or modified without authorization | Both knowledge endpoints were public; there was no API-key check. | Original `/knowledge` routes. |
| Conversation memory could grow without bounds | Conversation messages and conversation IDs were stored in process memory without size limits. | Original module-level `conversations` dictionary and append behavior. |
| User/business context was printed to logs | The request handler printed the business context and generated response. | Original `/chat` success branch. |
| Importing chatbot files could prompt for input or start API work | CLI/test code ran at module import time. | Original `chatbot.py`, `AI_service/ai_service.py`, and `AI_service/test_gemini.py`. |
| The public website chatbot did not answer the test question | The widget accepted “Which cloud platforms and AI services does XICTEK offer?” but showed no assistant reply after 40 seconds; the controls remained disabled. | Direct browser test on the public homepage. The website's server implementation is not part of this project, so the cause cannot be established here. |

## Changes implemented

- Added a shared XICTEK Systems/HisabDo system prompt and business context in `AI_service/ai_service.py`, with the homepage and services-page URLs as sources.
- Replaced the unrelated knowledge-base entries in `AI_service/chat_api.py` with website-backed company, services, AI, cloud, HisabDo, and contact entries.
- Added a minimum cosine-relevance cutoff and zero-vector handling. Retrieved results include their source URL.
- Encoded prompt context as JSON and instructed the model to treat submitted records/questions as data, not as instructions that override the grounding rules.
- Built conversation history from prior turns only, saved turns only after a successful model response, and bounded stored history to 20 messages per conversation and 1,000 conversations.
- Replaced success-shaped provider error results with sanitized HTTP 503 responses. Failures are logged server-side; request/business context is no longer printed.
- Protected `POST /knowledge` and `GET /knowledge` with `XICTEK_KNOWLEDGE_API_KEY`, using constant-time comparison. Knowledge changes continue to clear the embedding cache.
- Added request-size limits for chat and knowledge fields.
- Applied the shared verified prompt to `/ask` and made its provider errors sanitized HTTP 503 responses.
- Made the root CLI import-safe by delegating to the shared runner, and made `test_gemini.py` import-safe so test discovery does not start an interactive loop.
- Added `AI_service/test_chat_api.py` with deterministic tests that mock model and embedding calls.

## Tests and verification

| Check | Result |
|---|---|
| Python standard-library `unittest` discovery (`AI_service`, `test_*.py`) | **10 passed, 0 failed, 0 errors** |
| Pylance workspace diagnostics | **No diagnostics reported** |
| Pylance syntax checks on changed Python files | **No syntax errors** |
| FastAPI `TestClient` smoke check for `AI_service.app` home route | **HTTP 200** |
| FastAPI route registration smoke check for `AI_service.chat_api` | **Expected chat, knowledge, and conversation routes present** |
| Public website homepage and services-page review | **Completed; business facts above verified against page content** |
| Public XICTEK website chatbot | **Test question submitted; no response after 40 seconds and send controls remained disabled** |
| Live Gemini generation and live embedding API | **Not run**; tests use mocks and do not incur API calls |

## Remaining issues and recommendations

1. **Public website assistant outage:** investigate the live widget's browser/network/server logs. Its implementation is outside this repository, so the observed timeout is not fixed by these local changes.
2. **Conversation privacy/authentication:** `GET /conversations/{conversation_id}` and chat submission still need a real user authentication and tenant-authorization design before exposing the API to untrusted users. Conversation IDs alone are not an access-control boundary.
3. **Persistence and scaling:** conversation data is process-local and is lost on restart; use a durable store with retention and access policies if conversations must persist or multiple workers are deployed.
4. **Knowledge-key configuration:** set `XICTEK_KNOWLEDGE_API_KEY` in the deployment environment before using either knowledge endpoint. Knowledge endpoints intentionally return HTTP 503 when it is not configured.
5. **Retrieval threshold quality:** the `0.35` cosine cutoff is covered by a deterministic low-score test but has not been calibrated against live Gemini embeddings. Build a representative question/answer evaluation set before tuning it.
6. **Live model behavior:** run a controlled staging evaluation for supported questions, unknown facts, prompt injection, model/API outages, and follow-up memory. The local tests verify prompt contents and API behavior, not the model's final factuality.
7. **Data handling:** business/customer context is sent to the configured Gemini service when users call the chatbot. Confirm consent, data minimization, and retention requirements for the deployment.
