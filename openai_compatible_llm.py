"""
ComfyUI single-file custom node: OpenAI-compatible LLM caller

Install:
  1. Put this file into: ComfyUI/custom_nodes/comfyui_openai_compatible_llm_node.py
  2. Optional: pip install python-dotenv
  3. Put OPENAI_API_KEY=... in your environment or .env file when using OpenAI/OpenRouter/etc.
     Local servers such as LM Studio/Ollama can leave the key empty.
  4. Restart ComfyUI.

Examples:
  OpenAI:    api_base_url = https://api.openai.com/v1
  LM Studio: api_base_url = http://127.0.0.1:1234/v1
  Ollama:    api_base_url = http://127.0.0.1:11434/v1

This node intentionally uses only the Python standard library for HTTP calls.
python-dotenv is optional; if installed, load_dotenv() is called.

Optional unload_after_call supports provider-specific cleanup for LM Studio and Ollama.
Unload uses the same Authorization header as the main request.
Unload failures are returned as warnings in unload_json; the main LLM output is preserved.
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------------------------------------------------------
# dotenv support
# -----------------------------------------------------------------------------

def _fallback_load_dotenv(filename: str = ".env") -> None:
    """Tiny fallback .env reader used only when python-dotenv is not installed."""
    candidates = [
        os.path.join(os.getcwd(), filename),
        os.path.join(os.path.dirname(__file__), filename),
    ]
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
        except Exception:
            # Do not fail node import because .env parsing failed.
            pass


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        _fallback_load_dotenv()


_load_dotenv_if_available()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _normalize_chat_completions_url(api_base_url: str) -> str:
    """Accept either a base URL or a full /chat/completions endpoint."""
    url = (api_base_url or "").strip()
    if not url:
        raise ValueError("api_base_url is empty")

    url = url.rstrip("/")
    parsed = urllib.parse.urlparse(url)
    path = parsed.path.rstrip("/")

    if path.endswith("/chat/completions"):
        return url

    if path == "" or path == "/":
        suffix = "/v1/chat/completions"
    elif path.endswith("/v1"):
        suffix = "/chat/completions"
    else:
        suffix = "/chat/completions"

    return url + suffix


def _parse_json_object(value: str, field_name: str) -> Dict[str, Any]:
    text = (value or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"{field_name} must be a JSON object: {e}") from e
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return parsed


def _parse_stop(stop: str) -> Optional[List[str]]:
    text = (stop or "").strip()
    if not text:
        return None
    # Prefer JSON array when provided, otherwise use non-empty lines.
    if text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
            raise ValueError("stop must be a JSON string array or newline-separated strings")
        return parsed
    return [line for line in text.splitlines() if line]


def _image_tensor_to_data_urls(image: Any, max_images: int = 1) -> List[str]:
    """Convert ComfyUI IMAGE tensor [B,H,W,C], float 0..1, into PNG data URLs."""
    try:
        from PIL import Image
        import numpy as np
    except Exception as e:
        raise RuntimeError(
            "Pillow and NumPy are required for image input. They are normally available in ComfyUI."
        ) from e

    if image is None:
        raise ValueError("image is None")

    # ComfyUI IMAGE is normally torch.Tensor [B,H,W,C]. Accept numpy/list-like too.
    if hasattr(image, "detach"):
        arr = image.detach().cpu().numpy()
    else:
        arr = np.asarray(image)

    if arr.ndim == 3:
        arr = arr[None, ...]
    if arr.ndim != 4:
        raise ValueError(f"Expected IMAGE tensor with shape [B,H,W,C] or [H,W,C], got {arr.shape}")

    count = max(1, min(int(max_images or 1), int(arr.shape[0])))
    data_urls: List[str] = []

    for i in range(count):
        frame = np.clip(arr[i], 0.0, 1.0)
        frame = (frame * 255.0).round().astype("uint8")

        if frame.shape[-1] == 1:
            pil = Image.fromarray(frame[..., 0], mode="L")
        elif frame.shape[-1] == 3:
            pil = Image.fromarray(frame, mode="RGB")
        elif frame.shape[-1] == 4:
            pil = Image.fromarray(frame, mode="RGBA")
        else:
            raise ValueError(f"Expected 1, 3, or 4 image channels, got {frame.shape[-1]}")

        buffer = io.BytesIO()
        pil.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        data_urls.append(f"data:image/png;base64,{b64}")

    return data_urls


def _build_user_content(user_prompt: str, image: Any, image_detail: str, image_max_count: int) -> Any:
    if image is None:
        return user_prompt

    content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    detail = (image_detail or "auto").strip()

    for data_url in _image_tensor_to_data_urls(image, image_max_count):
        image_url_obj: Dict[str, Any] = {"url": data_url}
        # OpenAI Chat Completions vision currently documents: low, high, original, auto.
        # "omit" is kept for maximum compatibility with stricter local OpenAI-compatible servers.
        if detail != "omit":
            image_url_obj["detail"] = detail
        content.append({"type": "image_url", "image_url": image_url_obj})

    return content


def _extract_message_text(message: Any) -> str:
    if message is None:
        return ""
    if isinstance(message, str):
        return message
    if not isinstance(message, dict):
        return str(message)

    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def _extract_reasoning_text(choice: Dict[str, Any]) -> str:
    message = choice.get("message") or {}
    candidates = [
        message.get("reasoning_content"),
        message.get("reasoning"),
        message.get("thinking"),
        choice.get("reasoning"),
        choice.get("thinking"),
    ]
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, indent=2)
    return ""


def _apply_thinking_params(body: Dict[str, Any], thinking: str, thinking_api_style: str) -> None:
    """
    Apply optional reasoning/thinking controls.

    Important: there is no fully universal OpenAI-compatible thinking switch.
    Keep thinking_api_style='none' for maximum compatibility.
    """
    if thinking_api_style == "none" or thinking == "default":
        return

    # Map simple UX choices to effort levels.
    effort_map = {
        "off": "none",
        "on": "medium",
        "minimal": "minimal",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "xhigh",
    }
    effort = effort_map.get(thinking, "medium")

    if thinking_api_style == "reasoning_effort":
        body["reasoning_effort"] = effort
    elif thinking_api_style == "reasoning_object":
        body["reasoning"] = {"effort": effort}
    elif thinking_api_style == "ollama_think":
        if thinking == "off":
            body["think"] = False
        elif thinking == "on":
            body["think"] = True
        elif effort in ("low", "medium", "high"):
            body["think"] = effort
        else:
            body["think"] = True
    elif thinking_api_style == "reasoning_effort_and_object":
        body["reasoning_effort"] = effort
        body["reasoning"] = {"effort": effort}


def _origin_from_url(url: str) -> str:
    """Return scheme://host:port from a base or endpoint URL."""
    text = (url or "").strip().rstrip("/")
    if not text:
        raise ValueError("URL is empty")
    parsed = urllib.parse.urlparse(text)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid URL: {url}")
    return f"{parsed.scheme}://{parsed.netloc}"


def _detect_unload_provider(provider: str, api_base_url: str) -> str:
    """
    Best-effort provider detection.

    There is no standard OpenAI-compatible provider identity endpoint. We only use
    conservative URL/port heuristics and let the user override the result.
    """
    p = (provider or "auto").strip().lower()
    if p in ("none", "off", "disabled"):
        return "none"
    if p in ("lmstudio", "lm_studio", "lm-studio"):
        return "lmstudio"
    if p == "ollama":
        return "ollama"
    if p != "auto":
        return "none"

    parsed = urllib.parse.urlparse((api_base_url or "").strip())
    host = (parsed.hostname or "").lower()
    port = parsed.port
    path = (parsed.path or "").lower()

    # Common defaults: LM Studio OpenAI-compatible server is 1234; Ollama is 11434.
    if port == 1234:
        return "lmstudio"
    if port == 11434:
        return "ollama"

    # Mild hints for non-standard reverse-proxy paths.
    if "lmstudio" in host or "lm-studio" in host or "lmstudio" in path or "lm-studio" in path:
        return "lmstudio"
    if "ollama" in host or "ollama" in path:
        return "ollama"

    return "none"


def _build_common_headers(api_key_env: str, extra_headers_json: str = "") -> Dict[str, str]:
    headers: Dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "ComfyUI-OpenAI-Compatible-LLM-Node/1.0",
    }
    env_name = (api_key_env or "").strip()
    api_key = os.environ.get(env_name, "") if env_name else ""
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers_json:
        headers.update({str(k): str(v) for k, v in _parse_json_object(extra_headers_json, "extra_headers_json").items()})
    return headers


def _get_json(url: str, headers: Dict[str, str], timeout_sec: int) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from LLM server: {err_body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to connect to LLM server: {e}") from e

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM server returned non-JSON response: {raw[:1000]}") from e


def _find_lmstudio_loaded_instance_id(models_payload: Dict[str, Any], model: str) -> Optional[str]:
    """Find a loaded LM Studio instance id that corresponds to the requested model."""
    requested = (model or "").strip()
    models = models_payload.get("models", [])
    if not isinstance(models, list):
        return None

    all_loaded_ids: List[str] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        loaded = item.get("loaded_instances", [])
        if not isinstance(loaded, list) or not loaded:
            continue
        ids = [inst.get("id") for inst in loaded if isinstance(inst, dict) and isinstance(inst.get("id"), str)]
        all_loaded_ids.extend(ids)

        keys_to_match = [
            item.get("key"),
            item.get("display_name"),
            item.get("selected_variant"),
        ]
        variants = item.get("variants")
        if isinstance(variants, list):
            keys_to_match.extend(variants)

        # Exact loaded instance id wins.
        for instance_id in ids:
            if requested and instance_id == requested:
                return instance_id

        # Match by model key/variant/display name, then unload its first loaded instance.
        if requested and any(isinstance(k, str) and k == requested for k in keys_to_match):
            return ids[0] if ids else None

    # Helpful but conservative fallback: if only one model instance is loaded, it is probably the one
    # this just-called node used, even if the OpenAI-compatible model alias differs from LM Studio's key.
    if len(all_loaded_ids) == 1:
        return all_loaded_ids[0]

    return None


def _unload_lmstudio_model(
    api_base_url: str,
    model: str,
    headers: Dict[str, str],
    timeout_sec: int,
) -> Dict[str, Any]:
    """Unload a model in LM Studio via its Native REST API."""
    origin = _origin_from_url(api_base_url)
    endpoint = origin.rstrip("/") + "/api/v1/models/unload"

    # LM Studio's unload endpoint wants a loaded instance id. That id is not practical
    # for a ComfyUI workflow user to know, so always resolve it from /api/v1/models.
    models_endpoint = origin.rstrip("/") + "/api/v1/models"
    models_payload = _get_json(models_endpoint, headers, timeout_sec)
    instance_id = _find_lmstudio_loaded_instance_id(models_payload, model) or ""

    if not instance_id:
        raise ValueError(
            "Could not resolve LM Studio loaded instance id for this model. "
            "If multiple models are loaded, unload manually or make the model name match the LM Studio model key."
        )

    return _post_json(endpoint, {"instance_id": instance_id}, headers, timeout_sec)


def _unload_ollama_model(api_base_url: str, model: str, headers: Dict[str, str], timeout_sec: int) -> Dict[str, Any]:
    """Unload a model in Ollama via keep_alive=0."""
    origin = _origin_from_url(api_base_url)
    endpoint = origin.rstrip("/") + "/api/generate"
    model_name = (model or "").strip()
    if not model_name:
        raise ValueError("Ollama unload requires model")
    # Ollama documents unloading by sending an empty prompt and keep_alive=0.
    # stream=false avoids NDJSON streaming and keeps parsing simple.
    return _post_json(endpoint, {"model": model_name, "prompt": "", "keep_alive": 0, "stream": False}, headers, timeout_sec)


def _unload_model_after_call(
    provider: str,
    api_base_url: str,
    model: str,
    headers: Dict[str, str],
    timeout_sec: int,
) -> Tuple[str, Dict[str, Any]]:
    detected = _detect_unload_provider(provider, api_base_url)
    if detected == "lmstudio":
        return detected, _unload_lmstudio_model(api_base_url, model, headers, timeout_sec)
    if detected == "ollama":
        return detected, _unload_ollama_model(api_base_url, model, headers, timeout_sec)
    return detected, {"skipped": True, "reason": "No supported unload provider detected. Set unload_provider explicitly."}


def _post_json(url: str, body: Dict[str, Any], headers: Dict[str, str], timeout_sec: int) -> Dict[str, Any]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} from LLM server: {err_body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to connect to LLM server: {e}") from e

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM server returned non-JSON response: {raw[:1000]}") from e


# -----------------------------------------------------------------------------
# ComfyUI node
# -----------------------------------------------------------------------------

class OpenAICompatibleLLM:
    """Call an OpenAI-compatible /v1/chat/completions server and return text."""

    CATEGORY = "LLM/OpenAI Compatible"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "reasoning_text", "raw_json", "usage_json", "unload_json")
    OUTPUT_TOOLTIPS = (
        "The assistant's main text response extracted from the first choice.",
        "Reasoning/thinking text if the provider returns it separately; otherwise empty.",
        "The full JSON response from the chat completions endpoint.",
        "The usage object from the response, such as token counts, when provided.",
        "Provider-specific unload result or warning. Empty JSON when unload_after_call is off.",
    )
    DESCRIPTION = "Minimal OpenAI-compatible Chat Completions caller with optional IMAGE input and optional LM Studio/Ollama unload."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_base_url": (
                    "STRING",
                    {
                        "default": "https://api.openai.com/v1",
                        "multiline": False,
                        "tooltip": "Base URL or full /chat/completions endpoint. Examples: https://api.openai.com/v1, http://127.0.0.1:1234/v1, http://127.0.0.1:11434/v1.",
                    },
                ),
                "api_key_env": (
                    "STRING",
                    {
                        "default": "OPENAI_API_KEY",
                        "multiline": False,
                        "tooltip": "Environment variable that contains the Bearer API key/token. Leave empty for local servers that do not require authentication.",
                    },
                ),
                "model": (
                    "STRING",
                    {
                        "default": "gpt-4o-mini",
                        "multiline": False,
                        "tooltip": "Model ID to send to the API server. Use the exact name expected by OpenAI, LM Studio, Ollama, or your compatible server.",
                    },
                ),
                "system_prompt": (
                    "STRING",
                    {
                        "default": "You are a helpful assistant.",
                        "multiline": True,
                        "tooltip": "High-level behavior instruction sent as the system message. Leave empty to omit the system message.",
                    },
                ),
                "user_prompt": (
                    "STRING",
                    {
                        "default": "Describe the provided image in detail, focusing on visible subjects, composition, colors, lighting, style, and mood.",
                        "multiline": True,
                        "tooltip": "Main user request sent as the user message. When an IMAGE is connected, this text is sent together with the image.",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Controls randomness. Lower values are more deterministic; higher values are more varied. Usually adjust this or top_p, not both.",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Nucleus sampling cutoff. 1.0 disables the cutoff. Usually leave at 1.0 when tuning temperature.",
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 1,
                        "max": 262144,
                        "step": 64,
                        "tooltip": "Maximum number of tokens to generate. Some reasoning-capable servers may count hidden reasoning tokens against this budget.",
                    },
                ),
                "max_tokens_field": (
                    ["max_tokens", "max_completion_tokens", "both"],
                    {
                        "default": "max_tokens",
                        "tooltip": "Which token-limit field to send. max_tokens is most compatible; max_completion_tokens is newer OpenAI style; both is only for servers that accept both.",
                    },
                ),
                "thinking": (
                    ["default", "off", "on", "minimal", "low", "medium", "high", "xhigh"],
                    {
                        "default": "default",
                        "tooltip": "Reasoning/thinking preference. It is ignored unless thinking_api_style is set to a provider-specific style.",
                    },
                ),
                "thinking_api_style": (
                    [
                        "none",
                        "reasoning_effort",
                        "reasoning_object",
                        "reasoning_effort_and_object",
                        "ollama_think",
                    ],
                    {
                        "default": "none",
                        "tooltip": "Provider-specific way to send thinking controls. Use none for maximum compatibility. The wrong style may be rejected by some servers.",
                    },
                ),
            },
            "optional": {
                "image": (
                    "IMAGE",
                    {
                        "tooltip": "Optional ComfyUI IMAGE input. Images are encoded as PNG data URLs and attached to the user message.",
                    },
                ),
                "image_detail": (
                    ["auto", "low", "high", "original", "omit"],
                    {
                        "default": "auto",
                        "tooltip": "Vision detail hint for image understanding. auto/low/high/original follow OpenAI-style image input options; omit sends no detail field for stricter local servers.",
                    },
                ),
                "image_max_count": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 16,
                        "step": 1,
                        "tooltip": "Maximum number of images to send from a ComfyUI image batch. Start with 1 to avoid large requests.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 2147483647,
                        "step": 1,
                        "tooltip": "Optional sampling seed. -1 omits the seed. Determinism is best-effort and depends on the provider/model.",
                    },
                ),
                "presence_penalty": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Penalizes tokens that have already appeared, encouraging new topics. 0.0 is neutral and recommended by default.",
                    },
                ),
                "frequency_penalty": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": -2.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": "Penalizes repeated tokens based on how often they appear. Raise slightly if the model repeats phrases. 0.0 is neutral.",
                    },
                ),
                "stop": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Optional. Newline-separated stops or JSON array.",
                        "tooltip": "Optional stop sequences. Generation stops before any listed string is returned. Use newline-separated strings or a JSON array of strings.",
                    },
                ),
                "json_mode": (
                    ["off", "json_object"],
                    {
                        "default": "off",
                        "tooltip": "When json_object is selected, sends response_format={type: json_object}. The prompt should still explicitly ask for JSON.",
                    },
                ),
                "extra_body_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Optional JSON object merged into request body, e.g. {\"repetition_penalty\":1.05}",
                        "tooltip": "Advanced escape hatch. A JSON object merged into the request body after normal fields, so it can add or override provider-specific parameters.",
                    },
                ),
                "extra_headers_json": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "Optional JSON object merged into HTTP headers.",
                        "tooltip": "Advanced escape hatch. A JSON object merged into HTTP headers after the default Content-Type, Accept, User-Agent, and Authorization headers.",
                    },
                ),
                "unload_after_call": (
                    ["off", "on"],
                    {
                        "default": "off",
                        "tooltip": "If on, attempts to unload the model after the LLM response. Useful for freeing VRAM before downstream image generation.",
                    },
                ),
                "unload_provider": (
                    ["auto", "lmstudio", "ollama", "none"],
                    {
                        "default": "auto",
                        "tooltip": "Provider used for model unload. auto detects common LM Studio/Ollama ports; choose explicitly when using a custom host, port, or reverse proxy.",
                    },
                ),
                "timeout_sec": (
                    "INT",
                    {
                        "default": 120,
                        "min": 1,
                        "max": 3600,
                        "step": 1,
                        "tooltip": "HTTP timeout in seconds for the main request and optional unload calls. Increase if model loading or long responses time out.",
                    },
                ),
            },
        }

    def run(
        self,
        api_base_url: str,
        api_key_env: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        max_tokens_field: str,
        thinking: str,
        thinking_api_style: str,
        image: Any = None,
        image_detail: str = "auto",
        image_max_count: int = 1,
        seed: int = -1,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        stop: str = "",
        json_mode: str = "off",
        extra_body_json: str = "",
        extra_headers_json: str = "",
        unload_after_call: str = "off",
        unload_provider: str = "auto",
        timeout_sec: int = 120,
    ) -> Tuple[str, str, str, str, str]:
        endpoint = _normalize_chat_completions_url(api_base_url)

        messages: List[Dict[str, Any]] = []
        if (system_prompt or "").strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": _build_user_content(user_prompt, image, image_detail, image_max_count),
            }
        )

        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stream": False,
        }

        if max_tokens_field == "max_completion_tokens":
            body["max_completion_tokens"] = int(max_tokens)
        elif max_tokens_field == "both":
            body["max_tokens"] = int(max_tokens)
            body["max_completion_tokens"] = int(max_tokens)
        else:
            body["max_tokens"] = int(max_tokens)

        if presence_penalty != 0.0:
            body["presence_penalty"] = float(presence_penalty)
        if frequency_penalty != 0.0:
            body["frequency_penalty"] = float(frequency_penalty)
        if seed is not None and int(seed) >= 0:
            body["seed"] = int(seed)

        stop_values = _parse_stop(stop)
        if stop_values:
            body["stop"] = stop_values

        if json_mode == "json_object":
            body["response_format"] = {"type": "json_object"}

        _apply_thinking_params(body, thinking, thinking_api_style)

        # User-specified extra body wins, so advanced provider knobs can override defaults.
        body.update(_parse_json_object(extra_body_json, "extra_body_json"))

        headers = _build_common_headers(api_key_env, extra_headers_json)

        response = _post_json(endpoint, body, headers, int(timeout_sec))
        raw_json = json.dumps(response, ensure_ascii=False, indent=2)

        choices = response.get("choices") or []
        if not choices:
            # Some servers may return {message:{...}} or {response:"..."}; handle gently.
            text = _extract_message_text(response.get("message") or response.get("response") or response)
            reasoning_text = ""
        else:
            choice0 = choices[0]
            text = _extract_message_text(choice0.get("message"))
            reasoning_text = _extract_reasoning_text(choice0)

        usage_json = json.dumps(response.get("usage", {}), ensure_ascii=False, indent=2)

        unload_json = "{}"
        if (unload_after_call or "off").strip().lower() == "on":
            try:
                provider_name, unload_response = _unload_model_after_call(
                    unload_provider,
                    api_base_url,
                    model,
                    headers,
                    int(timeout_sec),
                )
                unload_json = json.dumps(
                    {"provider": provider_name, "response": unload_response},
                    ensure_ascii=False,
                    indent=2,
                )
            except Exception as e:
                # Warn-only by design: the LLM result is usually still useful, and ComfyUI/
                # downstream image-generation nodes can handle any remaining VRAM pressure.
                unload_json = json.dumps(
                    {"warning": True, "error": str(e), "policy": "warn"},
                    ensure_ascii=False,
                    indent=2,
                )

        return (text, reasoning_text, raw_json, usage_json, unload_json)


NODE_CLASS_MAPPINGS = {
    "OpenAICompatibleLLM": OpenAICompatibleLLM,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAICompatibleLLM": "OpenAI Compatible LLM",
}
