# evaluation/ragas_eval.py

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import json
import os
from typing import Any, Dict, List, Tuple

from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    context_precision,
    context_recall,   # se importan, pero NO se usan aún
    faithfulness,
    answer_relevancy,
)

from langchain_ollama import ChatOllama, OllamaEmbeddings

from src.backend.generator import generate_answer


# ============================================================
#  1. Ítems de prueba
# ============================================================

TEST_ITEMS: List[Dict[str, str]] = [
    {
        "question": "¿Cómo se llamaba el gato del cuento 'El gato negro'?",
        "ground_truth": "El gato del narrador se llamaba Plutón.",
    },
    {
        "question": "¿Dónde se posó el cuervo en el cuento 'El cuervo'?",
        "ground_truth": "El cuervo se posó sobre un busto de Minerva o Palas, sobre la puerta.",
    },
    {
        "question": "¿Qué relación tenía Leonora con el narrador en 'El cuervo'?",
        "ground_truth": "Leonora era la amada del narrador, cuya muerte le causó profunda desventura.",
    },
    {
        "question": "¿Qué órgano del cuerpo es central en el cuento 'El corazón delator'?",
        "ground_truth": "El órgano central del cuento es el corazón del anciano.",
    },
]


# ============================================================
#  2. Construir dataset a partir de tu RAG
# ============================================================

def build_eval_dataset() -> Tuple[Dataset, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []

    for item in TEST_ITEMS:
        question = item["question"]
        gt = item["ground_truth"]

        answer, docs = generate_answer(question)
        contexts = [d.page_content for d in docs] if docs else []

        rows.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": gt,
            }
        )

    dataset = Dataset.from_list(rows)
    return dataset, rows


# ============================================================
#  3. Juez phi4-mini en Ollama con limpieza de JSON
#     (ajustado SOLO para métricas tipo Verification,
#      p. ej. context_precision)
# ============================================================

class JsonStrictOllama(ChatOllama):
    """
    Variante de ChatOllama que:
    - Pide JSON al modelo (format="json" en __init__).
    - Limpia fences ```json ... ``` y markdown.
    - Asegura que la salida sea SIEMPRE un JSON válido del tipo:
        {"reason": <str>, "verdict": <0|1>}
      que es lo que espera el esquema Verification usado en context_precision.
    """

    @staticmethod
    def _strip_json_fences(text: str) -> str:
        s = text.strip()
        if s.startswith("```"):
            lines = s.splitlines()
            # quitar primera línea ```...
            lines = lines[1:]
            # quitar última línea si es ```
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            s = "\n".join(lines).strip()
        return s

    @staticmethod
    def _normalize_json(text: str) -> str:
        """
        Para context_precision (Verification) normalizamos TODO a:
            {"reason": <str>, "verdict": <0|1>}

        Estrategia:
        - Si no hay texto: JSON mínimo.
        - Si el parseo falla: tratamos toda la salida como "reason" y ponemos verdict=0.
        - Si hay JSON:
            * Si ya viene con "reason" y "verdict": se fuerzan tipos.
            * Si no, se encapsula como reason=str(data) y verdict=0.
        """

        s = text.strip()
        if not s:
            return json.dumps(
                {"reason": "empty response from model", "verdict": 0},
                ensure_ascii=False,
            )

        try:
            data = json.loads(s)
        except Exception:
            # JSON roto: usamos la cadena original como reason
            return json.dumps(
                {
                    "reason": s[:1000],
                    "verdict": 0,
                },
                ensure_ascii=False,
            )

        # Si ya viene en el formato esperado
        if isinstance(data, dict) and "reason" in data and "verdict" in data:
            return json.dumps(
                {
                    "reason": str(data["reason"]),
                    "verdict": int(data["verdict"]),
                },
                ensure_ascii=False,
            )

        # Cualquier otra estructura JSON: la usamos como reason genérico
        return json.dumps(
            {
                "reason": str(data)[:1000],
                "verdict": 0,
            },
            ensure_ascii=False,
        )

    def _postprocess_result(self, result):
        gens = result.generations

        # Aplanar lista de generaciones de forma defensiva
        flat_gens = []
        for item in gens:
            if isinstance(item, list):
                flat_gens.extend(item)
            else:
                flat_gens.append(item)

        for gen in flat_gens:
            # 1) Arreglar message.content si existe
            msg = getattr(gen, "message", None)
            if msg is not None:
                content = getattr(msg, "content", None)
                if isinstance(content, str):
                    cleaned = self._strip_json_fences(content)
                    cleaned = self._normalize_json(cleaned)
                    msg.content = cleaned

            # 2) Arreglar también gen.text si existe (RAGAS suele usar esto)
            text = getattr(gen, "text", None)
            if isinstance(text, str):
                cleaned_text = self._strip_json_fences(text)
                cleaned_text = self._normalize_json(cleaned_text)
                gen.text = cleaned_text

        return result

    def _generate(self, messages, stop=None, **kwargs):
        res = super()._generate(messages, stop=stop, **kwargs)
        return self._postprocess_result(res)

    async def _agenerate(self, messages, stop=None, **kwargs):
        res = await super()._agenerate(messages, stop=stop, **kwargs)
        return self._postprocess_result(res)


def get_ragas_models():
    """
    LLM juez: phi4-mini:3.8b en Ollama (formato JSON)
    Embeddings: embeddinggemma en Ollama
    """
    judge_base_url = os.getenv(
        "OLLAMA_EVAL_BASE_URL",
        "https://ec0041129340.ngrok-free.app",
    )
    judge_model_name = os.getenv("OLLAMA_EVAL_MODEL", "phi4-mini:3.8b")

    embed_base_url = os.getenv(
        "OLLAMA_EMBED_BASE_URL",
        "http://localhost:11434",
    )
    embed_model_name = os.getenv("OLLAMA_EMBED_MODEL", "embeddinggemma:latest")

    llm_judge = JsonStrictOllama(
        model=judge_model_name,
        base_url=judge_base_url,
        temperature=0.0,
        num_ctx=2048,
        num_predict=256,
        format="json",
        keep_alive=-1,
    )

    embeddings = OllamaEmbeddings(
        model=embed_model_name,
        base_url=embed_base_url,
    )

    return llm_judge, embeddings


# ============================================================
#  4. Ejecutar evaluación RAGAS (solo context_precision)
# ============================================================

def run_ragas_evaluation(dataset: Dataset) -> Dict[str, Any]:
    llm_judge, embeddings = get_ragas_models()

    # Por ahora solo context_precision
    metrics = [
        context_precision,
    ]

    for m in metrics:
        m.llm = llm_judge
        m.embeddings = embeddings

    run_config = RunConfig(
        timeout=600,
        max_workers=1,
    )

    print("\n=== Ejecutando todas las métricas con una sola llamada a RAGAS ===")
    res = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm_judge,
        embeddings=embeddings,
        raise_exceptions=True,
        run_config=run_config,
    )

    df = res.to_pandas()
    all_results: Dict[str, Any] = {}

    for m in metrics:
        name = m.name
        all_results[name] = {
            "per_sample": df[name].tolist(),
            "mean": float(df[name].mean()),
        }

    return all_results


# ============================================================
#  5. Guardar JSONs y punto de entrada
# ============================================================

def main(
    dataset_json_path: str = "evaluation/ragas_eval_dataset.json",
    summary_json_path: str = "evaluation/ragas_eval_summary.json",
):
    dataset, rows = build_eval_dataset()

    with open(dataset_json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    results = run_ragas_evaluation(dataset)

    metrics_summary: Dict[str, Any] = {
        "n_samples": len(rows),
        "metrics": {name: vals["mean"] for name, vals in results.items()},
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=2)

    print(f"\nDataset de evaluación guardado en: {dataset_json_path}")
    print(f"Resumen de métricas guardado en:  {summary_json_path}")
    print("Resumen:", metrics_summary)


if __name__ == "__main__":
    main()
