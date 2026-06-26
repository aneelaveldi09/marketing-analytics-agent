import os
from typing import Generator
from openai import OpenAI
from src.embeddings import embed_query
from src.vector_store import search
from src.config import GROK_MODEL, TOP_K

SYSTEM_PROMPT = """You are a Marketing Analytics AI Assistant built by Aneela Veldi, a Business Analytics graduate from DePaul University.

You are an expert in: campaign analytics, attribution modeling, A/B testing, customer segmentation, churn prediction, influencer ROI, marketing mix modeling, customer lifetime value, funnel analysis, and business intelligence.

Use the knowledge base context below to ground your answer. Be specific, practical, and business-focused. When referencing data or definitions, cite the source topic naturally (e.g. "According to research on attribution modeling..."). Use bullet points and structure when it helps clarity.

If the context does not fully cover the question, clearly say so and supplement with your general expertise.

Knowledge base context:
{context}
"""

FOLLOWUP_PROMPT = """Based on this conversation, suggest exactly 3 short follow-up questions the user might want to ask next.
Return only the 3 questions, one per line, no numbering, no dashes, no extra text."""


def _client() -> OpenAI:
    key = os.getenv("XAI_API_KEY")
    if not key:
        raise ValueError("XAI_API_KEY not set in .env")
    return OpenAI(api_key=key, base_url="https://api.x.ai/v1")


def retrieve(question: str, k: int = TOP_K) -> tuple[list[dict], list[float]]:
    query_emb = embed_query(question)
    return search(query_emb, k=k)


def ask_stream(question: str, history: list[dict], chunks: list[dict]) -> Generator[str, None, None]:
    context = "\n\n---\n\n".join(
        f"[{c['source']}]\n{c['text']}" for c in chunks
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        *history[-8:],
        {"role": "user", "content": question},
    ]
    stream = _client().chat.completions.create(
        model=GROK_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=1400,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def get_followups(question: str, answer: str) -> list[str]:
    try:
        resp = _client().chat.completions.create(
            model=GROK_MODEL,
            messages=[
                {"role": "user",      "content": question},
                {"role": "assistant", "content": answer},
                {"role": "user",      "content": FOLLOWUP_PROMPT},
            ],
            temperature=0.4,
            max_tokens=120,
        )
        lines = [
            l.strip().lstrip("0123456789.-) ")
            for l in resp.choices[0].message.content.strip().splitlines()
            if l.strip()
        ]
        return lines[:3]
    except Exception:
        return []
