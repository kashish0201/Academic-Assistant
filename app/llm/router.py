from enum import Enum
from typing import List

from openai import AzureOpenAI

from app.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
)
from app.llm.generator import is_greeting


class RetrievalRoute(str, Enum):
    DIRECT = "direct"
    LOCAL = "local"
    WEB = "web"
    HYBRID = "hybrid"


class QueryRouter:
    def __init__(self):
        self._client = None
        self.deployment = AZURE_OPENAI_DEPLOYMENT

    @property
    def client(self) -> AzureOpenAI:
        if self._client is None:
            self._client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
            )
        return self._client

    def route(self, query: str, history: List[dict] | None = None) -> RetrievalRoute:
        if is_greeting(query):
            return RetrievalRoute.DIRECT

        history = history or []
        history_text = ""
        for turn in history[-4:]:
            role = "Student" if turn["role"] == "user" else "Assistant"
            history_text += f"{role}: {turn['content'][:200]}\n"

        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You route student questions for a California university admissions assistant.\n"
                        "Pick exactly ONE route:\n"
                        "- DIRECT: greetings, thanks, small talk, or general conversation with no factual lookup\n"
                        "- LOCAL: answer is likely in uploaded policy PDFs (general CSU/UC transfer rules, "
                        "system-wide deadlines, GPA requirements in knowledge base)\n"
                        "- WEB: campus-specific facts, internships, program pages, rankings, comparisons, "
                        "research reputation, or any data not in uploaded PDFs\n"
                        "- HYBRID: needs both uploaded documents and live web (compare system policy with a campus)\n\n"
                        "Reply with only one word: DIRECT, LOCAL, WEB, or HYBRID."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Recent conversation:\n{history_text or 'None'}\n\n"
                        f"Question: {query}"
                    ),
                },
            ],
            temperature=0.0,
        )

        label = response.choices[0].message.content.strip().upper()
        for route in RetrievalRoute:
            if route.value.upper() in label or route.name in label:
                return route
        return RetrievalRoute.HYBRID
