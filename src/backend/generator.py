# src/backend/generator.py

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from .retriever import get_ensemble_retriever

# ---------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if not GOOGLE_API_KEY:
    raise RuntimeError("Falta GOOGLE_API_KEY en el .env")

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    api_key=GOOGLE_API_KEY,
    temperature=0.2,
    max_output_tokens=2048,
)


# ---------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------

def _build_context_block(docs: List[Document]) -> str:
    """
    Convierte la lista de documentos en un bloque de contexto legible,
    incluyendo metadatos básicos (source, chunk_id).
    """
    bloques: List[str] = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        source = meta.get("source", "desconocido")
        chunk_id = meta.get("chunk_id", meta.get("id", f"doc_{i}"))
        bloque = (
            f"[doc{i} | source={source} | chunk_id={chunk_id}]\n"
            f"{d.page_content}"
        )
        bloques.append(bloque)

    return "\n\n".join(bloques)


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Eres un asistente experto que responde en español.\n"
                "Debes contestar la pregunta del usuario usando EXCLUSIVAMENTE "
                "la información del contexto proporcionado.\n\n"
                "Si la respuesta no se puede obtener del contexto, indícalo de forma explícita "
                "y, si es útil, sugiere qué información adicional se requeriría.\n\n"
            ),
        ),
        (
            "human",
            "CONTEXTO:\n{context}\n\nPREGUNTA:\n{question}\n\nResponde en español.",
        ),
    ]
)


# ---------------------------------------------------------------------
# Chain RAG
# ---------------------------------------------------------------------

def build_rag_chain(k_candidates: int = 8):
    """
    input (str) -> retriever -> docs -> contexto
                 -> PROMPT -> LLM -> texto
    """
    retriever = get_ensemble_retriever(k=k_candidates)

    rag_chain = (
        {
            "context": retriever | RunnableLambda(_build_context_block),
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()  # convierte AIMessage -> str
    )
    return rag_chain, retriever


def generate_answer(
    question: str,
    k: int = 5,
    k_candidates: int = 8,
) -> Tuple[str, List[Document]]:
    """
    1) Usa un chain RAG (retriever + prompt + LLM) para generar la respuesta.
    2) Recupera también los documentos usados (top-k del ensemble).

    Devuelve:
        (respuesta, documentos_utilizados)
    """
    rag_chain, retriever = build_rag_chain(k_candidates=k_candidates)

    # 1) Respuesta usando el chain completo (ya es str)
    answer = rag_chain.invoke(question)
    if isinstance(answer, str):
        answer = answer.strip()
    else:
        # Fallback defensivo por si el parser no se aplicara
        answer = str(getattr(answer, "content", answer)).strip()

    # 2) Documentos (los mismos candidatos que se usan para el contexto)
    candidates = retriever.invoke(question)
    docs = candidates[:k] if candidates else []

    if not docs and not answer:
        return "No se encontraron fragmentos relevantes en la base de conocimiento.", []

    return answer, docs


# ---------------------------------------------------------------------
# Ejemplo de uso desde terminal
# ---------------------------------------------------------------------

def demo(question: str = "¿quién era Leonora?") -> None:
    answer, docs = generate_answer(
        question=question,
        k=5,
        k_candidates=8,
    )

    print("\n=== PREGUNTA ===")
    print(question)

    print("\n=== RESPUESTA (Gemini) ===")
    print(answer)

    print("\n=== CONTEXTO UTILIZADO ===")
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        src = meta.get("source", "desconocido")
        chunk_id = meta.get("chunk_id", meta.get("id", f"doc_{i}"))
        print(f"\n[doc{i}] source={src} | chunk_id={chunk_id}")
        print(d.page_content.replace("\n", " "))


if __name__ == "__main__":
    demo(question="¿Cómo murió la esposa del narrador del cuento del gato negro?")
