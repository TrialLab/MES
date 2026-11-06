"""
LLM registry for MES tool-layer calls, plus the MES agent-layer model.

Every tool-layer LLM call goes through ``LLMZoo.invoke(prompt) -> str`` or
``LLMZoo.stream(prompt) -> iterator of (kind, text)`` where ``kind`` is
``"text"`` for answer tokens and ``"reasoning"`` for thinking tokens.
Supports OpenAI (your own API key), LM Studio, Ollama and AWS Bedrock Claude backends.

Optional keyword arguments understood by the backends (ignored where unsupported):
    system       system prompt
    max_tokens   output token limit
    json_schema  JSON schema for constrained output (LM Studio)
    reasoning    False disables thinking on reasoning models (LM Studio)

Provider selection (environment variables or ``.env``):

    MES_LLM_PROVIDER      openai | lmstudio | bedrock           (default: openai)
    OPENAI_API_KEY        your OpenAI API key (also OPENAI_ORGANIZATION, OPENAI_PROJECT)
    MES_OPENAI_MODEL      OpenAI model                          (default: gpt-5.4-nano)
    OPENAI_BASE_URL       optional: another OpenAI-compatible endpoint for the openai backend
    LMSTUDIO_BASE_URL     LM Studio OpenAI-compatible endpoint  (default: http://localhost:1234/v1)
    LMSTUDIO_MODEL        model identifier loaded in LM Studio  (default: google/gemma-4-e4b)
    MES_BEDROCK_MODEL_ID  Bedrock model / inference profile     (default: us.anthropic.claude-opus-5)
    MES_BEDROCK_PLANNER_MODEL_ID  optional faster Bedrock model for the query planner (default: same model)

Bedrock uses the standard AWS credential chain (environment variables, ``AWS_PROFILE``, SSO, instance roles);
``AWS_REGION_NAME`` defaults to us-east-1. When the credentials have expired or are missing, the call fails with a
message saying how to log in again; ``MES_AWS_SSO_LOGIN=1`` runs ``aws sso login`` by itself instead. With
``MES_LLM_PROVIDER=lmstudio`` nothing touches AWS.

``MES_LLM_PROVIDER`` sets both the agent-layer model (``create_agent_model``)
and the default tool-layer backend (``create_llm()`` with no name).
"""

import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from mes import config  # noqa: F401  (loads .env and the HTTPS trust store)

logger = logging.getLogger(__name__)


class LLMZoo:
    def __init__(self):
        self.models = {}
        self.current_model = None
        self.current_model_name = None
        self.n_calls = 0
        self.disabled = {}       # helper model name -> error that made the main model take over
        self._lock = threading.Lock()

    def register_model(self, name, model_instance):
        self.models[name] = model_instance

    def select_model(self, name):
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found. Available: {list(self.models)}")
        self.current_model = self.models[name]
        self.current_model_name = name

    def _count(self):
        with self._lock:
            self.n_calls += 1

    def invoke(self, prompt, prefill=None, model=None, **kwargs):
        """
        ``model`` names a registered helper model for this call (e.g. "planner"); default: the selected model.
        A helper that fails (e.g. no access to it) is replaced by the selected model from then on.
        """
        if self.current_model is None:
            raise RuntimeError("No model selected.")
        self._count()
        helper = self.models.get(model) if model and model != self.current_model_name else None
        if helper is not None:
            try:
                return self._call(helper, prompt, prefill, **kwargs)
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                logger.warning("%s model failed (%s); using %s instead", model, error, self.current_model_name)
                if any(k in error for k in self._MODEL_ERRORS):   # this model cannot be used at all: stop trying
                    with self._lock:
                        self.models.pop(model, None)
                        self.disabled[model] = error
        return self._call(self.current_model, prompt, prefill, **kwargs)

    # Errors about the model itself (no access, unknown id), as opposed to transient network or service errors
    _MODEL_ERRORS = ("AccessDenied", "not authorized", "ValidationException", "ResourceNotFound", "NotFoundError")

    @staticmethod
    def _call(target, prompt, prefill, **kwargs):
        if prefill and getattr(target, "supports_prefill", False):
            return prefill + (target(prompt, prefill=prefill, **kwargs) or "")
        return target(prompt, **kwargs) or ""

    def stream(self, prompt, prefill=None, **kwargs):
        """
        Yield ``(kind, text)`` chunks; backends without streaming return the whole answer at once.
        ``prefill`` starts the assistant turn with that text on backends that support it (the
        prefill is yielded first, so the stream always contains the complete answer).
        """
        if self.current_model is None:
            raise RuntimeError("No model selected.")
        self._count()
        if prefill and getattr(self.current_model, "supports_prefill", False):
            yield "text", prefill
            kwargs["prefill"] = prefill
        if hasattr(self.current_model, "stream"):
            yield from self.current_model.stream(prompt, **kwargs)
        else:
            yield "text", self.current_model(prompt, **kwargs) or ""

    @staticmethod
    def _model_name(model):
        return getattr(model, "model", None) or getattr(model, "model_id", None) or getattr(model, "model_name", None)

    @property
    def label(self):
        name = self._model_name(self.current_model)
        label = f"{self.current_model_name} · {name}" if name and name != self.current_model_name else str(self.current_model_name)
        if "planner" in self.models:
            label += f" (planner: {self._model_name(self.models['planner'])})"
        return label

    def __repr__(self):
        return f"LLMZoo(current_model={self.current_model_name})"


def _strip_think(text):
    return text.split("</think>", 1)[1].strip() if "</think>" in text else text


class OllamaModel:
    """Local model served by Ollama."""

    def __init__(self, model="gemma3:4b", temperature=0.0, num_ctx=None, num_predict=None):
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_predict = num_predict

    def __call__(self, prompt, system=None, max_tokens=None, messages=None, **_):
        from ollama import chat
        options = {"temperature": self.temperature}
        if self.num_ctx:
            options["num_ctx"] = self.num_ctx
        if max_tokens or self.num_predict:
            options["num_predict"] = max_tokens or self.num_predict
        messages = ([{"role": "system", "content": system}] if system else []) + \
            (list(messages) if messages else [{"role": "user", "content": prompt}])
        response = chat(model=self.model, messages=messages, options=options)
        return _strip_think(response.message.content or "")


class BedrockClaudeModel:
    """AWS Bedrock Claude backend (boto3, standard AWS credential chain)."""

    def __init__(self, model_id="us.anthropic.claude-opus-5", region_name="us-east-1",
                 max_tokens=4096, temperature=0.0, profile_name=None):
        self.profile_name = profile_name
        self.region_name = region_name
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    @property
    def client(self):
        """bedrock-runtime client, created on first use (missing AWS credentials surface at call time)."""
        if self._client is None:
            import boto3
            session = boto3.Session(profile_name=self.profile_name, region_name=self.region_name)
            self._client = session.client("bedrock-runtime", config=self._config())
        return self._client

    @staticmethod
    def _config():
        """A reply with deep thinking can take minutes before its first byte (botocore waits 60 s by default and
        then retries the whole request several times); one retry, MES handles throttling itself."""
        from botocore.config import Config
        return Config(connect_timeout=10, read_timeout=300, retries={"max_attempts": 2, "mode": "standard"})

    @client.setter
    def client(self, value):
        self._client = value

    def _login_command(self):
        profile = self.profile_name or os.environ.get("AWS_PROFILE", "")
        return ["aws", "sso", "login"] + (["--profile", profile] if profile else [])

    def _refresh_client(self, error):
        """Expired or missing AWS credentials: say how to log in again — or, only with ``MES_AWS_SSO_LOGIN=1``, run
        ``aws sso login`` and rebuild the boto3 client."""
        import subprocess, boto3
        cmd = self._login_command()
        code = (getattr(error, "response", None) or {}).get("Error", {}).get("Code") or type(error).__name__
        hint = (f"AWS credentials for Bedrock have expired or are missing ({code}). Log in again — "
                f"for an SSO profile: {' '.join(cmd)} — or configure AWS credentials, then retry.")
        if os.environ.get("MES_AWS_SSO_LOGIN", "0").strip().lower() not in ("1", "true", "yes", "on"):
            raise RuntimeError(hint) from error
        print("\n[MES] AWS session expired; running: " + " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except Exception:
            raise RuntimeError(hint) from error
        session = boto3.Session(profile_name=self.profile_name, region_name=self.region_name)
        self.client = session.client("bedrock-runtime", config=self._config())

    # Models that still accept sampling parameters and have no effort control (Claude 3.x / 4.0 / 4.1 / 4.5 era)
    _LEGACY_MODELS = ("claude-3", "claude-instant", "sonnet-4-5", "haiku-4-5", "opus-4-1", "opus-4-0",
                      "sonnet-4-0", "sonnet-4-2025", "opus-4-2025")
    # Current models think adaptively by default; thinking tokens count against max_tokens
    THINKING_HEADROOM = 8000
    REFUSAL_NOTE = "\n\n_The model declined to answer this request._"
    TRUNCATION_NOTE = "\n\n_[answer truncated at the output limit]_"

    @property
    def _legacy(self):
        return any(tag in self.model_id for tag in self._LEGACY_MODELS)

    _effort_supported = True
    _cache_supported = True

    @staticmethod
    def effort(reasoning):
        """Effort level for ``reasoning``: False → low, None / True → medium, or an explicit level."""
        if reasoning in ("low", "medium", "high", "max"):
            return reasoning
        return "low" if reasoning is False else "medium"

    def _body(self, prompt, system=None, max_tokens=None, reasoning=None, messages=None, cache=False):
        max_tokens = max_tokens or self.max_tokens
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": list(messages) if messages else [{"role": "user", "content": prompt}],
        }
        if system:
            # A long system prompt reused over many calls (the analyst loop) is cached by Bedrock
            body["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}] \
                if cache and self._cache_supported else system
        if self._legacy:
            if self.temperature is not None:
                body["temperature"] = self.temperature
        else:
            # Current Claude models reject sampling parameters; thinking depth is set with effort instead
            if self._effort_supported:
                body["output_config"] = {"effort": self.effort(reasoning)}
            max_tokens += self.THINKING_HEADROOM
        body["max_tokens"] = max_tokens
        return json.dumps(body)

    # Bedrock refuses requests above the account's rate (several users, the analyst and its reviewer at once):
    # wait and try again instead of failing the answer
    THROTTLE_WAITS = (2, 5, 10)

    def _invoke(self, method, **kwargs):
        """Call Bedrock; if the account rejects ``output_config``, retry once without it; retry when throttled."""
        from botocore.exceptions import ClientError
        build = lambda: dict(modelId=self.model_id, body=self._body(**kwargs))
        throttled = 0
        for _attempt in range(3 + len(self.THROTTLE_WAITS)):
            try:
                return self._with_refresh(lambda: method(**build()))
            except ClientError as e:
                message = str(e)
                code = (getattr(e, "response", None) or {}).get("Error", {}).get("Code", "")
                if code in ("ThrottlingException", "ServiceUnavailableException", "ModelNotReadyException") \
                        and throttled < len(self.THROTTLE_WAITS):
                    time.sleep(self.THROTTLE_WAITS[throttled])
                    throttled += 1
                    continue
                if "ValidationException" not in message:
                    raise
                if self._effort_supported and any(k in message for k in ("output_config", "effort", "extraneous key")):
                    self._effort_supported = False
                elif self._cache_supported and kwargs.get("cache"):      # prompt caching not available here
                    self._cache_supported = False
                else:
                    raise
        return self._with_refresh(lambda: method(**build()))

    def _with_refresh(self, call):
        from botocore.exceptions import ClientError, NoCredentialsError
        for attempt in range(2):
            try:
                return call()
            except (ClientError, Exception) as e:
                # some errors carry response=None (a read timeout is also a requests exception)
                err_code = (getattr(e, "response", None) or {}).get("Error", {}).get("Code", "")
                if err_code in ("ExpiredTokenException", "UnrecognizedClientException") or "expired" in str(e).lower() \
                        or isinstance(e, NoCredentialsError):
                    if attempt == 0:
                        self._refresh_client(e)
                        continue
                raise

    def __call__(self, prompt, system=None, max_tokens=None, reasoning=None, messages=None, cache=False, **_):
        response = self._invoke(self.client.invoke_model, prompt=prompt, system=system, max_tokens=max_tokens,
                                reasoning=reasoning, messages=messages, cache=cache)
        result = json.loads(response["body"].read())
        text = "".join(block.get("text", "") for block in result.get("content", []) if block.get("type") == "text")
        if result.get("stop_reason") == "refusal":
            text += self.REFUSAL_NOTE
        return text

    def stream(self, prompt, system=None, max_tokens=None, reasoning=None, messages=None, cache=False, **_):
        response = self._invoke(self.client.invoke_model_with_response_stream, prompt=prompt, system=system,
                                max_tokens=max_tokens, reasoning=reasoning, messages=messages, cache=cache)
        body = response["body"]
        try:
            for event in body:
                chunk = json.loads(event["chunk"]["bytes"]) if "chunk" in event else {}
                kind = chunk.get("type")
                if kind == "content_block_delta":
                    delta = chunk.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield "text", delta.get("text", "")
                    elif delta.get("type") == "thinking_delta":
                        yield "reasoning", delta.get("thinking", "")
                elif kind == "message_delta":
                    stop = (chunk.get("delta") or {}).get("stop_reason")
                    if stop == "refusal":
                        yield "text", self.REFUSAL_NOTE
                    elif stop == "max_tokens":
                        yield "text", self.TRUNCATION_NOTE
        finally:
            if hasattr(body, "close"):
                body.close()


class LMStudioModel:
    """Local model served by LM Studio (OpenAI-compatible API)."""

    # Thinking models spend output tokens on reasoning before the answer
    REASONING_HEADROOM = 2048
    # LM Studio continues a trailing assistant message (used to start answers at a fixed heading)
    supports_prefill = True

    def __init__(self, model=None, base_url=None, temperature=0.0, max_tokens=4096):
        from openai import OpenAI
        self.model = model or lmstudio_model()
        self.temperature = temperature
        self.max_tokens = max_tokens
        # LM Studio ignores the API key, but the OpenAI client requires one
        self.client = OpenAI(base_url=base_url or lmstudio_base_url(), api_key="lm-studio", timeout=600)
        self._reasoning_control = True

    def _request(self, prompt, system=None, max_tokens=None, json_schema=None, reasoning=None, stream=False,
                 prefill=None, messages=None, cache=False):
        messages = ([{"role": "system", "content": system}] if system else []) + \
            (list(messages) if messages else [{"role": "user", "content": prompt}])
        max_tokens = max_tokens or self.max_tokens
        kwargs = dict(model=self.model, messages=messages, temperature=self.temperature, stream=stream)
        if prefill:
            messages.append({"role": "assistant", "content": prefill})
            # Generating the prefill again means the model started repeating its answer
            kwargs["stop"] = [prefill.strip()]
        if json_schema:
            kwargs["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "output", "strict": True, "schema": json_schema}}
        if reasoning in (False, "low") and self._reasoning_control:
            kwargs["extra_body"] = {"reasoning_effort": "none"}
        elif reasoning not in (False, "low"):
            max_tokens += self.REASONING_HEADROOM
        kwargs["max_tokens"] = max_tokens
        try:
            return self.client.chat.completions.create(**kwargs)
        except Exception as e:
            # Models without a reasoning switch may reject the parameter; retry once without it
            if "extra_body" in kwargs and "reasoning" in str(e).lower():
                self._reasoning_control = False
                kwargs.pop("extra_body")
                return self.client.chat.completions.create(**kwargs)
            raise

    # A small local server: several long requests at once can exhaust memory and crash the model
    _slots = threading.BoundedSemaphore(max(1, int(os.environ.get("MES_LMSTUDIO_PARALLEL", "2"))))
    CRASHED = "The model has crashed"

    def _call_with_retry(self, prompt, **kwargs):
        """One retry after a crash (LM Studio reloads the model on the next request)."""
        for attempt in range(2):
            try:
                return self._request(prompt, **kwargs)
            except Exception as e:
                if attempt == 0 and self.CRASHED in str(e):
                    logger.warning("LM Studio model crashed; retrying once: %s", e)
                    time.sleep(8)
                    continue
                raise

    def __call__(self, prompt, **kwargs):
        with self._slots:
            response = self._call_with_retry(prompt, **kwargs)
        return _strip_think(response.choices[0].message.content or "")

    def stream(self, prompt, **kwargs):
        self._slots.acquire()
        try:
            response = self._call_with_retry(prompt, stream=True, **kwargs)
        except Exception:
            self._slots.release()
            raise
        try:
            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if reasoning:
                    yield "reasoning", reasoning
                if delta.content:
                    yield "text", delta.content
        finally:
            response.close()  # closing the HTTP stream stops generation on the server
            self._slots.release()


class OpenAIModel(LMStudioModel):
    """
    OpenAI chat-completions backend with your own API key (``OPENAI_API_KEY``, optionally ``OPENAI_ORGANIZATION`` /
    ``OPENAI_PROJECT``; ``OPENAI_BASE_URL`` for another OpenAI-compatible service). Request parameters a model does not
    accept (a temperature other than the default, ``max_completion_tokens``, a JSON schema) are dropped once it says
    so, and the call is made again.
    """

    supports_prefill = False            # the chat API does not continue a trailing assistant message
    REASONING_HEADROOM = 4096           # reasoning models spend output tokens before the answer
    _slots = threading.BoundedSemaphore(max(1, int(os.environ.get("MES_OPENAI_PARALLEL", "8"))))

    def __init__(self, model=None, temperature=0.0, max_tokens=4096):
        from openai import OpenAI
        self.model = model or openai_model()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(**openai_client_options(), timeout=600)
        self._unsupported = set()

    def _request(self, prompt, system=None, max_tokens=None, json_schema=None, reasoning=None, stream=False,
                 prefill=None, messages=None, cache=False):
        messages = ([{"role": "system", "content": system}] if system else []) + \
            (list(messages) if messages else [{"role": "user", "content": prompt}])
        limit = (max_tokens or self.max_tokens) + self.REASONING_HEADROOM
        kwargs = dict(model=self.model, messages=messages, stream=stream)
        if "temperature" not in self._unsupported:
            kwargs["temperature"] = self.temperature
        kwargs["max_tokens" if "max_completion_tokens" in self._unsupported else "max_completion_tokens"] = limit
        if json_schema and "response_format" not in self._unsupported:
            kwargs["response_format"] = {"type": "json_schema",
                                         "json_schema": {"name": "output", "strict": False, "schema": json_schema}}
        for _attempt in range(3):
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as e:
                text = str(e).lower()
                if "temperature" in kwargs and "temperature" in text:
                    kwargs.pop("temperature")
                    self._unsupported.add("temperature")
                elif "max_completion_tokens" in kwargs and "max_completion_tokens" in text:
                    kwargs["max_tokens"] = kwargs.pop("max_completion_tokens")
                    self._unsupported.add("max_completion_tokens")
                elif "response_format" in kwargs and ("response_format" in text or "json_schema" in text):
                    kwargs.pop("response_format")        # the callers parse JSON from plain text as well
                    self._unsupported.add("response_format")
                else:
                    raise
        return self.client.chat.completions.create(**kwargs)


OLLAMA_MODELS = {
    "gemma3": "gemma3:4b",
    "gemma4": "gemma4:e4b",
    "medgemma4b": "hf.co/unsloth/medgemma-4b-it-GGUF:latest",
}

# MES_LLM_PROVIDER value → LLMZoo name of its tool-layer backend
PROVIDER_LLM_NAMES = {"openai": "openai", "lmstudio": "lmstudio", "bedrock": "claude-bedrock"}


def get_provider():
    """Configured MES provider (``MES_LLM_PROVIDER``)."""
    provider = os.environ.get("MES_LLM_PROVIDER", "openai").strip().lower()
    if provider not in PROVIDER_LLM_NAMES:
        raise ValueError(f"Unknown MES_LLM_PROVIDER '{provider}'. Available: {list(PROVIDER_LLM_NAMES)}")
    return provider


def openai_model():
    return os.environ.get("MES_OPENAI_MODEL", "gpt-5.4-nano").strip() or "gpt-5.4-nano"


def openai_client_options():
    """Keyword arguments of the OpenAI clients: the API key and, when set, organization, project and base URL. A
    missing key fails at the first call (with OpenAI's message), not when MES starts."""
    options = {"api_key": os.environ.get("OPENAI_API_KEY") or "missing-OPENAI_API_KEY"}
    for env, key in (("OPENAI_ORGANIZATION", "organization"), ("OPENAI_PROJECT", "project"),
                     ("OPENAI_BASE_URL", "base_url")):
        if os.environ.get(env, "").strip():
            options[key] = os.environ[env].strip()
    return options


def lmstudio_base_url():
    return os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")


def lmstudio_model():
    return os.environ.get("LMSTUDIO_MODEL", "google/gemma-4-e4b")


def bedrock_model_id():
    return os.environ.get("MES_BEDROCK_MODEL_ID", "us.anthropic.claude-opus-5")


def create_agent_model(provider=None):
    """Build the Agents SDK model for the MES agents and orchestrator."""
    provider = provider or get_provider()
    if provider == "openai":
        from openai import AsyncOpenAI
        from agents import OpenAIChatCompletionsModel
        return OpenAIChatCompletionsModel(model=openai_model(), openai_client=AsyncOpenAI(**openai_client_options()))
    if provider == "lmstudio":
        from openai import AsyncOpenAI
        from agents import OpenAIChatCompletionsModel
        client = AsyncOpenAI(base_url=lmstudio_base_url(), api_key="lm-studio")
        return OpenAIChatCompletionsModel(model=lmstudio_model(), openai_client=client)
    # Bedrock Claude via LiteLLM (standard AWS credential chain) — imported lazily so LM Studio setups don't need it
    from agents.extensions.models.litellm_model import LitellmModel
    os.environ.setdefault("AWS_REGION_NAME", "us-east-1")
    return LitellmModel(model=f"bedrock/{bedrock_model_id()}")


def create_llm(model_name=None):
    """Build an LLMZoo with the requested backend selected.

    ``model_name`` defaults to the backend of ``MES_LLM_PROVIDER``.
    """
    model_name = model_name or PROVIDER_LLM_NAMES[get_provider()]
    llm = LLMZoo()
    if model_name == "claude-bedrock":
        profile = os.environ.get("AWS_PROFILE")
        region = os.environ.get("AWS_REGION_NAME", "us-east-1")
        llm.register_model("claude-bedrock", BedrockClaudeModel(
            model_id=bedrock_model_id(), region_name=region, max_tokens=4096, temperature=0.0, profile_name=profile,
        ))
        # Optional faster model for the query planner (a short JSON task), e.g. us.anthropic.claude-haiku-5-5
        planner_id = os.environ.get("MES_BEDROCK_PLANNER_MODEL_ID", "").strip()
        if planner_id and planner_id != bedrock_model_id():
            llm.register_model("planner", BedrockClaudeModel(
                model_id=planner_id, region_name=region, max_tokens=2048, temperature=0.0, profile_name=profile,
            ))
    elif model_name == "openai" or model_name.startswith("gpt"):
        llm.register_model(model_name, OpenAIModel(model=None if model_name == "openai" else model_name))
    elif model_name == "lmstudio":
        llm.register_model("lmstudio", LMStudioModel())
    else:
        llm.register_model(model_name, OllamaModel(
            model=OLLAMA_MODELS.get(model_name, model_name),
            temperature=0.0, num_ctx=65536, num_predict=4096,
        ))
    llm.select_model(model_name)
    return llm


def llm_map(func, items, max_workers=8):
    """Apply ``func`` to every item with parallel LLM calls."""
    items = list(items)
    if max_workers <= 1 or len(items) <= 1:
        return [func(item) for item in items]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(func, items))
