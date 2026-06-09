"""Streamlit UI — entrada principal do app. Pronta para deploy 1-click no Streamlit Cloud.

Voce nao precisa editar quase nada aqui — ja faz integracao com:
- src.pipeline.rag (TODOs 1-3)
- src.pipeline.cache (TODO 5)
- src.pipeline.routing (TODO 6)
- src.pipeline.tools (TODO 4, opcional)
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

# Adiciona o root do projeto no path para imports
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

load_dotenv()

import streamlit as st  # noqa: E402

from src.observability.trace import trace, log_event  # noqa: E402
from src.pipeline.cache import ExactCache, SemanticCache  # noqa: E402
from src.pipeline.rag import build_rag_pipeline  # noqa: E402
from src.pipeline.routing import classify_complexity  # noqa: E402


# ---------------------------------------------------------------- Streamlit UI
st.set_page_config(page_title="LGPD Copilot — Compliance", page_icon="🛡️", layout="centered")

# CSS customizado para visual premium e harmônico (Dark Glassmorphism)
st.markdown("""
<style>
    /* Estilização sutil de fundo e fontes */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #e2e8f0;
    }
    
    /* Forçar cores do corpo do texto */
    .stApp p, .stApp span, .stApp label, .stMarkdown p, div[data-testid="stMarkdownContainer"] p {
        color: #cbd5e1 !important;
    }
    
    h1 {
        color: #38bdf8 !important;
        font-weight: 800;
        text-shadow: 0 0 10px rgba(56, 189, 248, 0.2);
    }
    
    h2, h3, h4, h5, h6 {
        color: #38bdf8 !important;
    }
    
    /* Input customizado */
    .stTextInput input {
        border-radius: 10px !important;
        border: 1.5px solid #38bdf8 !important;
        background-color: rgba(15, 23, 42, 0.6) !important;
        color: #f8fafc !important;
        padding: 12px 18px !important;
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.1) !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput input:focus {
        border-color: #f43f5e !important;
        box-shadow: 0 4px 18px rgba(244, 63, 94, 0.2) !important;
        background-color: rgba(15, 23, 42, 0.8) !important;
    }
    
    /* Cards de métricas elegantes */
    div[data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-weight: 700;
    }
    
    div[data-testid="metric-container"] {
        background-color: rgba(30, 41, 59, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }
    
    /* Estilizar o painel lateral (Sidebar) */
    section[data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    section[data-testid="stSidebar"] h2 {
        color: #38bdf8 !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ LGPD Copilot")
st.caption("Assistente Inteligente de Compliance Baseado na Lei Geral de Proteção de Dados (Lei 13.709/2018)")


# Inicializacao lazy de pipeline + caches
@st.cache_resource
def get_pipeline():
    return build_rag_pipeline(corpus_dir=str(_ROOT / "data" / "corpus"))


@st.cache_resource
def get_exact_cache():
    return ExactCache()


@st.cache_resource
def get_semantic_cache():
    return SemanticCache(threshold=0.93)


with st.spinner("Inicializando pipeline RAG..."):
    pipeline = get_pipeline()
    exact_cache = get_exact_cache()
    semantic_cache = get_semantic_cache()


# Sidebar — metricas e debug
with st.sidebar:
    st.header("Metricas do Sistema")
    st.metric("Chunks no Vector Store", pipeline.collection.count())
    st.metric("Tamanho Exact Cache", exact_cache.stats()["size"])
    st.metric("Tamanho Semantic Cache", semantic_cache.stats()["size"])

    st.markdown("---")
    if st.button("Limpar caches"):
        try:
            get_exact_cache()._store.clear()
            get_semantic_cache()._queries.clear()
            get_semantic_cache()._embeddings.clear()
            get_semantic_cache()._answers.clear()
        except Exception:
            pass
        get_exact_cache.clear()
        get_semantic_cache.clear()
        st.success("Caches limpos com sucesso!")


# Initialize session state for the query if not present
if "query_input" not in st.session_state:
    st.session_state.query_input = ""

# Sugestões de perguntas para demonstração
st.markdown("##### 💡 Perguntas Sugeridas")
col_sug1, col_sug2, col_sug3 = st.columns(3)

if col_sug1.button("CPF sem consentimento?", use_container_width=True):
    st.session_state.query_input = "Posso tratar o CPF sem consentimento para faturamento de compras?"
if col_sug2.button("Controlador vs Operador", use_container_width=True):
    st.session_state.query_input = "Explique a diferenca entre controlador e operador de dados na LGPD."
if col_sug3.button("Direitos do titular", use_container_width=True):
    st.session_state.query_input = "Quais sao os direitos do titular definidos no Artigo 18?"

# Main — chat interface
query = st.text_input(
    "Sua pergunta:", 
    key="query_input",
    placeholder="Pergunte algo sobre a lei ou cenários de compliance..."
)

if query:
    with trace("query_handle", query=query) as ctx:
        trace_id = ctx["trace_id"]

        # 1. Exact cache
        cached = exact_cache.get(query)
        if cached:
            st.success("⚡ Cache Hit (Exato)")
            st.write(cached)
            log_event("cache_hit", trace_id=trace_id, layer="exact")
            st.stop()

        # 2. Semantic cache
        try:
            cached = semantic_cache.get(query)
        except NotImplementedError:
            cached = None
            st.warning("Semantic cache nao implementado (TODO 5). Caindo no LLM real.")

        if cached:
            st.success("🧠 Cache Hit (Semantico)")
            st.write(cached)
            log_event("cache_hit", trace_id=trace_id, layer="semantic")
            st.stop()

        # 3. Pipeline RAG + Routing
        model_to_use = None
        try:
            decision = classify_complexity(query)
            model_to_use = decision.model
            st.info(f"🚦 Roteador de Modelos: complexidade **{decision.complexity.upper()}** -> direcionado para o modelo **{decision.model}**")
            log_event("route_decision", trace_id=trace_id, **decision.__dict__)
        except NotImplementedError:
            st.warning("Routing nao implementado (TODO 6). Usando modelo default.")

        with st.spinner("Analisando documentos e gerando resposta..."):
            try:
                result = pipeline.answer(query, model=model_to_use)
            except NotImplementedError as e:
                st.error(f"Pipeline nao implementado: {e}")
                st.info("Implemente TODOs 1-3 em `src/pipeline/rag.py` para destravar.")
                st.stop()

        # 4. Renderiza resposta principal
        st.write(result["answer"])
        
        # Exibe fontes RAG e ferramenta customizada lado a lado
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            if result.get("sources"):
                with st.expander("📄 Fontes Citadas (ChromaDB RAG)"):
                    # Filtra fontes únicas
                    unique_sources = sorted(list(set(result["sources"])))
                    for src, pg in unique_sources:
                        st.write(f"- `{src}`: página {pg}")
                        
        with col_res2:
            if result.get("tool_calls"):
                with st.expander("🛠️ Chamada de Ferramenta (Function-calling)"):
                    for tc in result["tool_calls"]:
                        st.write(f"**Tool:** `{tc['name']}`")
                        st.write(f"**Args:** `{tc['arguments']}`")
                        st.write("**Resultado:**")
                        st.code(tc['result'], language="text")

        # Insere nos caches
        exact_cache.put(query, result["answer"])
        semantic_cache.put(query, result["answer"])
        log_event("answer_generated", trace_id=trace_id, sources=len(result.get("sources", [])))


st.divider()
st.caption(
    "Projeto Final — Disciplina: Desenvolvendo Software com IA Generativa (Mod4 / PPI) | "
    "Desenvolvido por Pietro Esteves"
)
