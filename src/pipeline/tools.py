"""Function-calling / tool-use — registro de tools usadas pelo agente.

Reaproveita o LAB-001. Voce vai preencher 1 TODO aqui (sua tool especifica).
"""

from __future__ import annotations

import json
from typing import Any, Callable


# ============================================================================
# TODO 4 — Sua tool especifica do dominio
# ============================================================================
# Cada projeto precisa de UMA tool customizada que faca sentido para o problema.
# Exemplos por dominio:
#   - Livro tecnico:    lookup_chapter(chapter: int) -> str
#   - Changelog:        check_compat(lib: str, version: str) -> dict
#   - Podcast:          get_timestamp(quote: str) -> str
#   - Codigo:           run_snippet(code: str) -> str  (sandboxed)
#   - Documentos legais: cite_article(law: str, article: int) -> str
#
# 1. Implemente a funcao Python real abaixo (substitua o exemplo)
# 2. Adicione o schema JSON em TOOLS abaixo
# 3. Registre em TOOL_REGISTRY
# ============================================================================


# SEU CODIGO AQUI — TODO 4
def cite_article(article_number: int) -> str:
    """Retorna o texto oficial e integral do Artigo N da LGPD.

    A funcao recebe o numero do artigo e retorna a string com o texto exato
    do artigo extraido diretamente do PDF oficial da LGPD.
    """
    import re
    from pathlib import Path
    from pypdf import PdfReader

    pdf_path = Path("data/corpus/LGPD.pdf")
    if not pdf_path.exists():
        return "ERROR: O arquivo oficial da lei (LGPD.pdf) nao foi encontrado na pasta data/corpus/."

    try:
        reader = PdfReader(pdf_path)
        full_text = ""
        for page in reader.pages:
            raw_text = page.extract_text() or ""
            text = re.sub(r'/uni([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), raw_text)
            text = re.sub(r'[ \t\xa0]+', ' ', text)
            full_text += text + "\n"

        # Padrao flexivel ancorado ao inicio da linha para encontrar "Art. N" original, ignorando citacoes cruzadas no meio do texto
        pattern = rf"\n\s*Art\.\s*{article_number}(?:º)?(?!\d)"
        match = re.search(pattern, full_text, re.IGNORECASE)

        if not match:
            return f"AVISO: O Artigo {article_number} nao foi localizado no texto da lei."

        start_idx = match.start()

        # Procura o inicio do proximo artigo de forma ancorada para limitar a resposta
        next_article = article_number + 1
        next_pattern = rf"\n\s*Art\.\s*{next_article}(?:º)?(?!\d)"
        next_match = re.search(next_pattern, full_text[start_idx + 10:], re.IGNORECASE)

        if next_match:
            end_idx = start_idx + 10 + next_match.start()
            article_text = full_text[start_idx:end_idx].strip()
        else:
            # Se nao achar o proximo, retorna ate 1500 caracteres da lei a partir do artigo encontrado
            article_text = full_text[start_idx:start_idx + 1500].strip()

        return article_text
    except Exception as e:
        return f"ERROR ao buscar o Artigo {article_number}: {e}"


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "cite_article",
            "description": "Retorna o texto oficial e integral de um artigo específico da LGPD (Lei 13.709/2018). Use para citar a lei e evitar alucinacoes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "article_number": {
                        "type": "integer",
                        "description": "O número do artigo da lei (ex: 7 para o Artigo 7º)."
                    }
                },
                "required": ["article_number"]
            }
        }
    }
]


TOOL_REGISTRY: dict[str, Callable[..., str]] = {
    "cite_article": cite_article,
}


def run_tool_call(name: str, arguments_json: str) -> str:
    """Executa uma tool call e retorna o resultado como string."""
    if name not in TOOL_REGISTRY:
        return f"ERROR: tool '{name}' nao registrada"
    try:
        kwargs = json.loads(arguments_json)
        return TOOL_REGISTRY[name](**kwargs)
    except Exception as e:
        return f"ERROR ao executar {name}: {e}"
