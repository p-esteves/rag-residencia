"""Cache em 2 niveis: exact-match (SHA256) + semantic (cosine similarity).

Reaproveita o notebook 05. Voce vai preencher 1 TODO aqui.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

import numpy as np
from openai import OpenAI


class ExactCache:
    """Cache por hash SHA256 da query. Captura replays exatos (~10-15% das queries).
    
    TTL: Volátil na RAM (limpo se o container reiniciar ou se inativo por ~1h).
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    @staticmethod
    def _key(query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()

    def get(self, query: str) -> str | None:
        return self._store.get(self._key(query))

    def put(self, query: str, answer: str) -> None:
        self._store[self._key(query)] = answer

    def stats(self) -> dict[str, int]:
        return {"size": len(self._store)}


class SemanticCache:
    """Cache por similaridade de embedding. Captura parafrases (~20% adicional).
    
    TTL: Volátil na RAM (limpo se o container reiniciar ou se inativo por ~1h).
    """

    def __init__(self, threshold: float = 0.93) -> None:
        self.threshold = threshold
        self._queries: list[str] = []
        self._embeddings: list[np.ndarray] = []
        self._answers: list[str] = []

        # Inicializa cliente para embeddings (mesmo provider do RAG)
        has_gemini = "GEMINI_API_KEY" in os.environ and "substitua" not in os.environ["GEMINI_API_KEY"]
        has_openai = "OPENAI_API_KEY" in os.environ and "substitua" not in os.environ["OPENAI_API_KEY"]
        
        if has_gemini:
            self._client = OpenAI(
                api_key=os.environ["GEMINI_API_KEY"],
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            self._embed_model = os.environ.get("EMBED_MODEL", "gemini-embedding-001")
            self._embed_fn = self._embed_openai
        elif has_openai:
            self._client = OpenAI()
            self._embed_model = "text-embedding-3-small"
            self._embed_fn = self._embed_openai
        else:
            from sentence_transformers import SentenceTransformer
            self._local_model = SentenceTransformer("all-MiniLM-L6-v2")
            self._embed_fn = self._embed_local

    def _embed_openai(self, text: str) -> np.ndarray:
        r = self._client.embeddings.create(model=self._embed_model, input=text)
        return np.array(r.data[0].embedding)

    def _embed_local(self, text: str) -> np.ndarray:
        return self._local_model.encode(text)

    def _embed(self, text: str) -> np.ndarray:
        return self._embed_fn(text)

    # ------------------------------------------------------------------ TODO 5
    def get(self, query: str) -> str | None:
        """Retorna resposta cacheada se similar a query alguma anterior, OU None."""
        if not self._queries:
            return None

        # SEU CODIGO AQUI — TODO 5
        # 1. Embedar a query (self._embed)
        e = self._embed(query)
        norm_e = np.linalg.norm(e)
        if norm_e == 0:
            return None

        # 2. Calcular similaridade cosseno contra todos self._embeddings:
        #    cos_sim = np.dot(e, em) / (np.linalg.norm(e) * np.linalg.norm(em))
        similarities = []
        for em in self._embeddings:
            norm_em = np.linalg.norm(em)
            if norm_em == 0:
                sim = 0.0
            else:
                sim = np.dot(e, em) / (norm_e * norm_em)
            similarities.append(sim)

        # 3. Pegar idx do maior; se sims[idx] >= self.threshold, retornar self._answers[idx]
        idx = int(np.argmax(similarities))
        if similarities[idx] >= self.threshold:
            return self._answers[idx]

        # 4. Caso contrario, retornar None
        return None

    def put(self, query: str, answer: str) -> None:
        self._queries.append(query)
        self._embeddings.append(self._embed(query))
        self._answers.append(answer)

    def stats(self) -> dict[str, Any]:
        return {"size": len(self._queries), "threshold": self.threshold}
