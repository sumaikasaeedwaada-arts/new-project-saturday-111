import hmac
import json
import logging
import os
from collections import OrderedDict
from datetime import datetime

import numpy as np
from fastapi import FastAPI, Header, HTTPException
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from AI_service.ai_service import BUSINESS_CONTEXT, SYSTEM_PROMPT

app = FastAPI()
logger = logging.getLogger(__name__)
MAX_HISTORY_MESSAGES = 20
MAX_CONVERSATIONS = 1000
MIN_RELEVANCE_SCORE = 0.35
client = genai.Client()

conversations = OrderedDict()

knowledge_base = [
    {
        "id": 1,
        "title": "XICTEK Systems",
        "content": (
            "XICTEK Systems is a product-focused software company building "
            "practical digital products and custom solutions."
        ),
        "source": "https://www.xicteksystems.com/"
    },
    {
        "id": 2,
        "title": "Software Development Services",
        "content": (
            "Services include custom software, web applications, mobile apps, "
            "and SaaS development. The services page describes web applications "
            "using React, Next.js, and TypeScript, and cross-platform mobile "
            "development using Flutter and React Native."
        ),
        "source": "https://www.xicteksystems.com/services"
    },
    {
        "id": 3,
        "title": "AI and Automation",
        "content": (
            "XICTEK Systems describes AI services including OpenAI and LLM "
            "integrations, chatbots, custom ML models, intelligent automation, "
            "and practical AI-assisted workflows."
        ),
        "source": "https://www.xicteksystems.com/services"
    },
    {
        "id": 4,
        "title": "Cloud Services",
        "content": (
            "XICTEK Systems describes cloud architecture, migration, and "
            "managed infrastructure on Azure and AWS, as well as DevOps and "
            "cloud infrastructure services."
        ),
        "source": "https://www.xicteksystems.com/services"
    },
    {
        "id": 5,
        "title": "HisabDo",
        "content": (
            "HisabDo is XICTEK Systems' expense-management and digital-ledger "
            "product for individuals and small businesses. The website lists "
            "offline-first records, digital khata and customer records, PDF "
            "reports, voice entry, and backup and restore."
        ),
        "source": "https://www.xicteksystems.com/"
    },
    {
        "id": 6,
        "title": "Contact",
        "content": (
            "The XICTEK Systems website lists xicteksystems@gmail.com as a "
            "contact address."
        ),
        "source": "https://www.xicteksystems.com/"
    }
]
vector_store = []


def create_embedding(text, task_type):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type)
    )
    return result.embeddings[0].values


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0
    return np.dot(a, b) / denominator


def build_vector_store():
    vector_store.clear()
    for item in knowledge_base:
        vector_store.append({
            "id": item["id"],
            "title": item["title"],
            "content": item["content"],
            "source": item["source"],
            "embedding": create_embedding(
                item["content"],
                "RETRIEVAL_DOCUMENT"
            )
        })


def search_knowledge(question):
    if not vector_store:
        build_vector_store()

    query_embedding = create_embedding(question, "RETRIEVAL_QUERY")
    results = []
    for item in vector_store:
        score = cosine_similarity(query_embedding, item["embedding"])
        results.append({
            "id": item["id"],
            "title": item["title"],
            "content": item["content"],
            "source": item["source"],
            "score": float(score)
        })

    results.sort(key=lambda item: item["score"], reverse=True)
    return [
        item for item in results[:3]
        if item["score"] >= MIN_RELEVANCE_SCORE
    ]


class ChatRequest(BaseModel):
    conversation_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=4000)
    user: str = Field(max_length=1000)
    business: str = Field(max_length=2000)
    customer: str = Field(max_length=2000)
    task: str = Field(max_length=2000)
    notes: str = Field(max_length=4000)


class KnowledgeRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=4000)


@app.get("/")
def home():
    return {"message": "XICTEK Systems and HisabDo AI Chat API is running"}


@app.post("/knowledge")
def add_knowledge(
    request: KnowledgeRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    require_knowledge_api_key(x_api_key)

    new_item = {
        "id": len(knowledge_base) + 1,
        "title": request.title,
        "content": request.content,
        "source": "Admin-provided knowledge"
    }
    knowledge_base.append(new_item)
    vector_store.clear()
    return {
        "message": "Knowledge added successfully",
        "knowledge": new_item
    }


@app.get("/knowledge")
def get_knowledge(
    x_api_key: str | None = Header(default=None, alias="X-API-Key")
):
    require_knowledge_api_key(x_api_key)
    return {"knowledge_base": knowledge_base}


def require_knowledge_api_key(provided_api_key: str | None) -> None:
    expected_api_key = os.getenv("XICTEK_KNOWLEDGE_API_KEY")
    if not expected_api_key:
        raise HTTPException(
            status_code=503,
            detail="Knowledge updates are not configured."
        )
    if not provided_api_key or not hmac.compare_digest(
        provided_api_key,
        expected_api_key
    ):
        raise HTTPException(status_code=401, detail="Invalid API key.")


@app.post("/chat")
def chat(request: ChatRequest):
    history = conversations.get(request.conversation_id, [])
    prompt_data = {
        "verified_business_context": BUSINESS_CONTEXT,
        "business_context": {
            "user": request.user,
            "business": request.business,
            "customer": request.customer,
            "task": request.task,
            "notes": request.notes
        },
        "conversation_history": history[-10:],
        "current_question": request.message
    }

    try:
        prompt_data["retrieved_knowledge"] = search_knowledge(request.message)
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=json.dumps(prompt_data, ensure_ascii=False),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT
            )
        )
        if not response.text:
            raise ValueError("Gemini returned an empty response.")
    except Exception as exc:
        logger.exception("Chat generation failed")
        raise HTTPException(
            status_code=503,
            detail=(
                "The AI service is temporarily unavailable. "
                "Please try again later."
            )
        ) from exc

    timestamp = datetime.now().isoformat()
    history.extend([
        {"role": "user", "message": request.message, "time": timestamp},
        {"role": "assistant", "message": response.text, "time": timestamp}
    ])
    conversations[request.conversation_id] = history[-MAX_HISTORY_MESSAGES:]
    conversations.move_to_end(request.conversation_id)
    while len(conversations) > MAX_CONVERSATIONS:
        conversations.popitem(last=False)

    return {
        "conversation_id": request.conversation_id,
        "question": request.message,
        "retrieved_knowledge": prompt_data["retrieved_knowledge"],
        "response": response.text
    }


@app.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    return {
        "conversation_id": conversation_id,
        "messages": conversations.get(conversation_id, [])
    }
