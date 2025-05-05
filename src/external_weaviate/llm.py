from typing import List

from dataclasses import dataclass
from openai import OpenAI


@dataclass
class ChatLLM:
    model: str
    base_url=None
    temperature=0.1
    thought_seperator: str = "</think>"
    api_key="placeholder"

    @property
    def client(self)->OpenAI:
        return OpenAI(base_url=self.base_url, api_key=self.api_key)
    def __init__(
        self,
        model,
        base_url=None,
        temperature=0.1,
        thought_seperator: str = "</think>",
    ):
        self.base_url = base_url
        self.model = model
        self.default_temperature = temperature
        self.thought_seperator = thought_seperator

    def call(self, prompt: str, temperature=None):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature or self.default_temperature,
        )
        answer = (
            response.choices[0]
            .message.content.split(self.thought_seperator)[-1]
            .strip()
        )
        return answer

    def __call__(self, prompt: str, temperature=None):
        return self.call(prompt, temperature)

class EmbeddingLLM:
    def __init__(self, model, base_url=None, api_key=""):
        self.base_url = base_url
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def call(self, inputs: List[str]) -> List[List[float]]:
        if isinstance(inputs, str):
            inputs = [inputs]
        response = self.client.embeddings.create(model=self.model, input=inputs)
        embeddings = [item.embedding for item in response.data]
        return embeddings
