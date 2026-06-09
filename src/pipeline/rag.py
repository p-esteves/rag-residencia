"""RAG pipeline — chunk, embed, index, retrieve, generate.

Reaproveita as funcoes do notebook 02. Voce vai preencher 3 TODOs aqui.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI


def _make_client() -> tuple[OpenAI, str | None]:
    """Inicializa cliente OpenAI-compatible conforme provider escolhido no .env."""
    if "GEMINI_API_KEY" in os.environ and "substitua" not in os.environ["GEMINI_API_KEY"]:
        client = OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        embed_api_base = "https://generativelanguage.googleapis.com/v1beta/openai/"
    elif "OPENAI_API_KEY" in os.environ and "substitua" not in os.environ["OPENAI_API_KEY"]:
        client = OpenAI()
        embed_api_base = None
    elif "GROQ_API_KEY" in os.environ and "substitua" not in os.environ["GROQ_API_KEY"]:
        client = OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )
        embed_api_base = None
    else:
        # Fallback para local / Ollama
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "dummy"),
            base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
        )
        embed_api_base = None
    return client, embed_api_base


class RAGPipeline:
    """Pipeline RAG end-to-end com Chroma local."""

    def __init__(
        self,
        corpus_dir: str = "data/corpus",
        persist_dir: str = "data/chroma",
        collection_name: str = "docs",
        llm_model: str | None = None,
        embed_model: str | None = None,
    ) -> None:
        self.client, embed_api_base = _make_client()
        
        # Determina o modelo padrão conforme a chave configurada
        if llm_model:
            self.llm_model = llm_model
        elif "GEMINI_API_KEY" in os.environ and "substitua" not in os.environ["GEMINI_API_KEY"]:
            self.llm_model = os.environ.get("LLM_MODEL", "gemini-2.5-flash-lite")
        elif "OPENAI_API_KEY" in os.environ and "substitua" not in os.environ["OPENAI_API_KEY"]:
            self.llm_model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        elif "GROQ_API_KEY" in os.environ and "substitua" not in os.environ["GROQ_API_KEY"]:
            self.llm_model = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
        else:
            self.llm_model = os.environ.get("LLM_MODEL", "qwen3:8b")

        self.embed_model = embed_model or os.environ.get("EMBED_MODEL", "gemini-embedding-001")

        # Configura a embedding function (OpenAI/Gemini vs Local SentenceTransformers)
        has_gemini = "GEMINI_API_KEY" in os.environ and "substitua" not in os.environ["GEMINI_API_KEY"]
        has_openai = "OPENAI_API_KEY" in os.environ and "substitua" not in os.environ["OPENAI_API_KEY"]
        
        if has_gemini or has_openai:
            embed_kwargs: dict[str, Any] = {
                "api_key": os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY"),
                "model_name": self.embed_model,
            }
            if embed_api_base:
                embed_kwargs["api_base"] = embed_api_base
            self.embed_fn = OpenAIEmbeddingFunction(**embed_kwargs)
        else:
            # Fallback para embeddings locais se estiver usando Groq puro ou Ollama
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            self.embed_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

        self.corpus_dir = Path(corpus_dir)
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        chroma = chromadb.PersistentClient(path=persist_dir)
        self.collection = chroma.get_or_create_collection(
            name=collection_name, embedding_function=self.embed_fn
        )

    # ------------------------------------------------------------------ TODO 1
    def ingest_and_index(self) -> int:
        """Le PDFs de `corpus_dir`, faz chunking e indexa em Chroma.

        Retorna numero de chunks indexados.

        Ja deixei a estrutura do ciclo. Voce completa as 3 partes marcadas.
        """
        from pypdf import PdfReader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # SEU CODIGO AQUI — TODO 1.A
        # Iterar por todos os PDFs em self.corpus_dir.
        # Para cada PDF, ler todas as paginas com PdfReader e extrair texto.
        # Acumular numa lista `docs` com dicts: {"text": str, "source": str, "page": int}
        # Dica: reaproveite o snippet do notebook 02 (Etapa 1 — Ingestao de PDFs).
        docs: list[dict] = []
        for pdf_path in sorted(self.corpus_dir.glob("*.pdf")):
            try:
                import re
                reader = PdfReader(pdf_path)
                for page_idx, page in enumerate(reader.pages):
                    raw_text = page.extract_text() or ""
                    text = re.sub(r'/uni([0-9a-fA-F]{4})', lambda m: chr(int(m.group(1), 16)), raw_text)
                    # Normaliza múltiplos espaços (incluindo espaços não-separáveis \xa0) para um único espaço para otimizar busca semântica
                    text = re.sub(r'[ \t\xa0]+', ' ', text)
                    text = re.sub(r' ?\n ?', '\n', text)
                    if text.strip():
                        docs.append({
                            "text": text,
                            "source": pdf_path.name,
                            "page": page_idx + 1
                        })
            except Exception as e:
                print(f"Erro ao ler PDF {pdf_path}: {e}")

        # SEU CODIGO AQUI — TODO 1.B
        # Aplicar RecursiveCharacterTextSplitter com chunk_size=800, overlap=100
        # Quebrar cada doc em chunks e construir lista `chunks` com:
        # {"id": unique_id, "text": str, "source": str, "page": int}
        # Dica: reaproveite o notebook 02 (Etapa 2 — Chunking Recursivo).
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
        chunks: list[dict] = []
        for d in docs:
            split_texts = splitter.split_text(d["text"])
            for i, text in enumerate(split_texts):
                chunk_id = f"{d['source']}-p{d['page']}-c{i}"
                chunks.append({
                    "id": chunk_id,
                    "text": text,
                    "source": d["source"],
                    "page": d["page"]
                })

        # SEU CODIGO AQUI — TODO 1.C
        # Adicionar chunks no Chroma via self.collection.add(ids=, documents=, metadatas=)
        # Lembre de filtrar metadatas para conter apenas {source, page} (Chroma rejeita listas).
        if chunks:
            # Divide os chunks em lotes de 80 para evitar o limite de 100 requests por batch da API do Gemini
            batch_size = 80
            for idx in range(0, len(chunks), batch_size):
                batch = chunks[idx : idx + batch_size]
                self.collection.add(
                    ids=[c["id"] for c in batch],
                    documents=[c["text"] for c in batch],
                    metadatas=[{"source": c["source"], "page": c["page"]} for c in batch]
                )

        return self.collection.count()

    # ------------------------------------------------------------------ TODO 2
    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """Busca top-k chunks similares a query."""
        # SEU CODIGO AQUI — TODO 2
        # Usar self.collection.query(query_texts=[query], n_results=k)
        # Retornar lista de dicts: {"text", "source", "page", "distance"}
        # Dica: notebook 02, Etapa 4 — Retrieval.
        result = self.collection.query(query_texts=[query], n_results=k)
        hits = []
        if result and result.get("documents") and result["documents"][0]:
            for i in range(len(result["documents"][0])):
                hits.append({
                    "text": result["documents"][0][i],
                    "source": result["metadatas"][0][i]["source"],
                    "page": result["metadatas"][0][i]["page"],
                    "distance": result["distances"][0][i] if result.get("distances") else 0.0,
                })
        return hits

    def answer(self, question: str, k: int = 5, model: str | None = None) -> dict:
        """Pipeline completo: retrieve + augment + generate + tool-use. Retorna {answer, sources}."""
        from src.pipeline.tools import TOOLS, run_tool_call

        hits = self.retrieve(question, k=k)

        # 1. Montar contexto concatenando os textos dos hits com cabecalho [source:page]
        context_parts = []
        for h in hits:
            context_parts.append(f"[{h['source']}:p{h['page']}]\n{h['text']}")
        context_str = "\n\n--\n\n".join(context_parts)

        # 2. Construir mensagens estruturadas (system e user roles) para guiar o LLM no uso de tools e RAG
        system_instruction = """Você é um assistente técnico especialista na LGPD. Responda à pergunta do usuário baseando-se no contexto legal fornecido.
Se a pergunta solicitar ou fizer menção direta a um artigo específico da lei (como o Artigo 18 ou Artigo 7º), você deve prioritariamente usar a ferramenta 'cite_article' para obter o texto oficial dele antes de responder.
Se a resposta não puder de forma alguma ser deduzida a partir do contexto fornecido ou das ferramentas, responda exatamente "Nao encontrado no corpus".
Você pode deduzir relações lógicas simples (por exemplo: CPF é um tipo de dado pessoal; faturamento de compras ou emissão de nota fiscal enquadram-se em execução de contrato ou cumprimento de obrigação legal).
Sempre cite a fonte no formato [arquivo:pagina]."""

        user_content = f"CONTEXTO:\n{context_str}\n\nPERGUNTA: {question}\n\nRESPOSTA:"

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_content}
        ]

        # Determina modelo a ser usado (propagando decisao do roteador se houver)
        model_to_use = model or self.llm_model

        # 3. Chamar self.client.chat.completions.create(model=model_to_use, ...) com suporte a tools
        response = self.client.chat.completions.create(
            model=model_to_use,
            messages=messages,
            tools=TOOLS if TOOLS else None,
            temperature=0.0
        )
        
        message = response.choices[0].message
        tool_calls_executed = []

        # 4. Processar possíveis tool calls
        if message.tool_calls:
            # Anexa o pedido do assistente de rodar a tool
            messages.append(message)
            
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args = tool_call.function.arguments
                result = run_tool_call(name, args)
                tool_calls_executed.append({
                    "name": name,
                    "arguments": args,
                    "result": result
                })
                # Anexa o resultado da tool como mensagem
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": result
                })

            # Segunda chamada ao LLM com o contexto da tool integrada
            second_response = self.client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                temperature=0.0
            )
            answer_text = second_response.choices[0].message.content or ""
        else:
            answer_text = message.content or ""

        sources = [(h["source"], h["page"]) for h in hits]

        return {
            "answer": answer_text,
            "sources": sources,
            "tool_calls": tool_calls_executed
        }


PROMPT_TEMPLATE = """Você é um assistente técnico especialista na LGPD. Responda à pergunta do usuário baseando-se no contexto legal fornecido abaixo.
Se a resposta não puder ser deduzida ou associada de forma alguma ao contexto fornecido, responda exatamente "Nao encontrado no corpus".
Você pode deduzir relações lógicas simples (por exemplo: CPF é um tipo de dado pessoal; faturamento de compras ou emissão de nota fiscal enquadram-se em execução de contrato ou cumprimento de obrigação legal).
Sempre cite a fonte usando o formato [arquivo:pagina].

CONTEXTO:
{context}

PERGUNTA: {question}

RESPOSTA:"""


def build_rag_pipeline(corpus_dir: str = "data/corpus") -> RAGPipeline:
    """Factory: cria pipeline e indexa corpus se ainda nao indexado."""
    pipeline = RAGPipeline(corpus_dir=corpus_dir)
    if pipeline.collection.count() == 0:
        pipeline.ingest_and_index()
    return pipeline
