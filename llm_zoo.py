"""
LLM registry used by all MES agents.

Every agent talks to a language model through a single object exposing
``invoke(prompt) -> str``. ``LLMZoo`` keeps a registry of backends (OpenAI and
local models served by Ollama) and forwards each call to the selected backend.
"""

import os
from concurrent.futures import ThreadPoolExecutor


class LLMZoo:
    def __init__(self):
        self.models = {}
        self.current_model = None
        self.current_model_name = None

    def register_model(self, name, model_instance):
        self.models[name] = model_instance

    def select_model(self, name):
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found. Please register it first.")
        self.current_model = self.models[name]
        self.current_model_name = name

    def invoke(self, prompt):
        """Run the selected model on ``prompt`` and return the response text."""
        if self.current_model is None:
            raise RuntimeError("No model selected. Please select a model first.")
        return self.current_model(prompt) or ""

    def __repr__(self):
        return f"LLMZoo(current_model={self.current_model_name})"


class OpenaiClient:
    """OpenAI chat-completions backend (e.g., GPT-5.4-nano)."""

    def __init__(self, api_key, organization=None, project=None, model_name="gpt-5.4-nano"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, organization=organization, project=project)
        self.model_name = model_name

    def __call__(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model_name, messages=[{"role": "user", "content": prompt}], temperature=0, top_p=1)
        return response.choices[0].message.content


class OllamaModel:
    """Local model served by Ollama (https://ollama.com)."""

    def __init__(self, model="gemma3:4b", temperature=0.0, num_ctx=None, num_predict=None):
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_predict = num_predict

    def __call__(self, prompt):
        from ollama import chat
        options = {"temperature": self.temperature}
        if self.num_ctx:
            options["num_ctx"] = self.num_ctx
        if self.num_predict:
            options["num_predict"] = self.num_predict
        response = chat(model=self.model, messages=[{"role": "user", "content": prompt}], options=options)
        text = response.message.content or ""
        # Reasoning models may prepend their chain of thought.
        if "</think>" in text:
            text = text.split("</think>", 1)[1].strip()
        return text


# Ollama backbones evaluated in the manuscript (served locally).
OLLAMA_MODELS = {
    "gemma3": "gemma3:4b",
    "gemma4": "gemma4:e4b",
    "medgemma4b": "hf.co/unsloth/medgemma-4b-it-GGUF:latest",
}


def create_llm(model_name="gpt-5.4-nano"):
    """
    Build an ``LLMZoo`` with the requested backbone selected.

    OpenAI models (names starting with ``gpt``) read ``OPENAI_API_KEY`` and the
    optional ``OPENAI_ORGANIZATION`` / ``OPENAI_PROJECT`` environment variables.
    Other names are resolved through ``OLLAMA_MODELS`` (or passed to Ollama as is).
    """
    llm = LLMZoo()
    if model_name.startswith("gpt"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set. Add it to your .env file or environment.")
        backend = OpenaiClient(
            api_key=api_key,
            organization=os.getenv("OPENAI_ORGANIZATION") or None,
            project=os.getenv("OPENAI_PROJECT") or None,
            model_name=model_name,
        )
    else:
        backend = OllamaModel(model=OLLAMA_MODELS.get(model_name, model_name), temperature=0.0,
                              num_ctx=65536, num_predict=4096)
    llm.register_model(model_name, backend)
    llm.select_model(model_name)
    return llm


def llm_map(func, items, max_workers=8):
    """
    Apply ``func`` to every item, running independent LLM calls in parallel.
    Results keep the input order. Use ``max_workers=1`` for local backends that
    serve one request at a time.
    """
    items = list(items)
    if max_workers <= 1 or len(items) <= 1:
        return [func(item) for item in items]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(func, items))
