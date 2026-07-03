from typing import Dict, List


class RAGEvaluator:
    """Optional RAG evaluation using Ragas. Install with: pip install ragas datasets"""

    def evaluate_interaction(
        self,
        question: str,
        answer: str,
        contexts: List[str],
    ) -> Dict[str, float]:
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import answer_relevancy, context_precision, faithfulness
        except ImportError as exc:
            raise ImportError(
                "Ragas evaluation requires optional dependencies: pip install ragas datasets"
            ) from exc

        dataset = Dataset.from_dict(
            {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }
        )

        results = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )

        return {key: float(value) for key, value in results.items()}
