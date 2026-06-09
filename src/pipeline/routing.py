"""Model routing cheap-first com fallback.

Reaproveita o notebook 05. Voce vai preencher 1 TODO aqui.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI


@dataclass(frozen=True)
class RouteDecision:
    model: str
    complexity: str  # "simple" | "complex"
    reason: str


# ------------------------------------------------------------------ TODO 6
def classify_complexity(query: str) -> RouteDecision:
    """Classifica complexidade da query para escolher modelo (cheap vs premium).

    Estrategia heuristica simples. Em producao, evoluiria para classifier treinado.
    """
    cheap_model = os.environ.get("CHEAP_MODEL")
    premium_model = os.environ.get("PREMIUM_MODEL")
    
    if not cheap_model:
        if "GEMINI_API_KEY" in os.environ and "substitua" not in os.environ["GEMINI_API_KEY"]:
            cheap_model = "gemini-2.5-flash-lite"
        elif "OPENAI_API_KEY" in os.environ and "substitua" not in os.environ["OPENAI_API_KEY"]:
            cheap_model = "gpt-4o-mini"
        elif "GROQ_API_KEY" in os.environ and "substitua" not in os.environ["GROQ_API_KEY"]:
            cheap_model = "llama-3.1-8b-instant"
        else:
            cheap_model = "qwen3:8b"

    if not premium_model:
        if "GEMINI_API_KEY" in os.environ and "substitua" not in os.environ["GEMINI_API_KEY"]:
            premium_model = "gemini-2.5-pro"
        elif "OPENAI_API_KEY" in os.environ and "substitua" not in os.environ["OPENAI_API_KEY"]:
            premium_model = "gpt-4o"
        elif "GROQ_API_KEY" in os.environ and "substitua" not in os.environ["GROQ_API_KEY"]:
            premium_model = "qwen/qwen3-32b"
        else:
            premium_model = "qwen3:8b"

    query_lower = query.lower().strip()
    
    # Palavras-chave que indicam complexidade analítica em português
    complex_keywords = [
        "explique", "compare", "analise", "projete", "como funciona",
        "diferenca", "relacao", "quais sao", "quais as", "limites",
        "consequencia", "impacto", "caso", "cenario", "se eu"
    ]

    is_complex = False
    reason = "Query curta ou direta classificada como simples por padrao."

    if any(kw in query_lower for kw in complex_keywords):
        is_complex = True
        reason = "Contem termos analiticos complexos (explique/compare/diferenca/etc.)."
    elif len(query) >= 80:
        is_complex = True
        reason = "Comprimento da query (>= 80 caracteres) sugere complexidade analitica."
    elif len(query) < 50 and query_lower.endswith("?"):
        is_complex = False
        reason = "Pergunta direta curta terminada em interrogacao."

    if is_complex:
        return RouteDecision(model=premium_model, complexity="complex", reason=reason)
    else:
        return RouteDecision(model=cheap_model, complexity="simple", reason=reason)


def make_client() -> OpenAI:
    """Cliente OpenAI-compatible para o provider configurado."""
    if "GEMINI_API_KEY" in os.environ and "substitua" not in os.environ["GEMINI_API_KEY"]:
        return OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    elif "OPENAI_API_KEY" in os.environ and "substitua" not in os.environ["OPENAI_API_KEY"]:
        return OpenAI()
    elif "GROQ_API_KEY" in os.environ and "substitua" not in os.environ["GROQ_API_KEY"]:
        return OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )
    # Fallback local / Ollama
    return OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
        base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
    )
