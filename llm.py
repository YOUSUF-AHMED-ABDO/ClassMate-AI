"""اتصال موحّد بـ Claude وChatGPT وGemini عن طريق HTTP عادي (من غير أي مكتبة إضافية).

الدالة الوحيدة اللي بتهمك: complete(cfg, system, user, json_mode)."""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

# الأسماء دي ممكن تتغير مع الوقت: المستخدم يقدر يعدّل اسم الموديل من الواجهة.
PROVIDERS = {
    "gemini": {
        "label": "Google Gemini",
        "model": "gemini-3.8-flash",
        "env": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "key_url": "https://aistudio.google.com/app/apikey",
        "key_hint": "AIza...",
        "free": "فيه طبقة مجانية (محدودة السرعة)",
    },
    "openai": {
        "label": "OpenAI (ChatGPT)",
        "model": "gpt-5.4-nano",
        "env": ["OPENAI_API_KEY"],
        "key_url": "https://platform.openai.com/api-keys",
        "key_hint": "sk-...",
        "free": "مدفوع بالاستخدام (أقل رصيد حوالي 5 دولار)",
    },
    "anthropic": {
        "label": "Anthropic (Claude)",
        "model": "claude-haiku-5-5",
        "env": ["ANTHROPIC_API_KEY"],
        "key_url": "https://console.anthropic.com/settings/keys",
        "key_hint": "sk-ant-...",
        "free": "مدفوع بالاستخدام",
    },
}


@dataclass
class LLMConfig:
    provider: str
    api_key: str
    model: str = ""

    @property
    def model_name(self) -> str:
        return self.model.strip() or PROVIDERS[self.provider]["model"]


class LLMError(Exception):
    """رسالة خطأ مفهومة بالعربي تتعرض للمستخدم. suggested = اسم موديل بديل اقترحه المزوّد (لو موجود)."""

    suggested: str = ""


def _post(url: str, headers: dict, body: dict, timeout: int = 120) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers}, method="POST")
    delay = 4
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(raw)
                detail = detail.get("error", detail)
                detail = detail.get("message", detail) if isinstance(detail, dict) else detail
            except Exception:
                detail = raw[:200]
            if e.code in (429, 500, 502, 503, 529) and attempt < 3:
                time.sleep(delay)
                delay *= 2
                continue
            err = LLMError(_explain(e.code, str(detail)))
            m = re.search(r"(?:use|try|switch to)\s+(?:models/)?([\w.\-]+)", str(detail), re.I)
            if m and ("no longer available" in str(detail) or e.code == 404):
                err.suggested = m.group(1)
                err.args = (err.args[0] + f" ← جرّب الموديل: {err.suggested}",)
            raise err from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt < 3:
                time.sleep(delay)
                delay *= 2
                continue
            raise LLMError(f"مفيش اتصال بالإنترنت أو الخدمة مش بترد ({e}).") from None
    raise LLMError("فشل الاتصال بعد عدة محاولات.")


def _explain(code: int, detail: str) -> str:
    detail = detail[:300]
    if code in (401, 403):
        return f"المفتاح (API key) غلط أو مش مفعّل أو معندوش صلاحية. ({detail})"
    if code == 404:
        return f"اسم الموديل غلط أو مش متاح لحسابك، جرّب اسم تاني. ({detail})"
    if code == 429:
        return f"وصلت للحد الأقصى (كوتة أو سرعة الطلبات)، أو الرصيد خلص. استنى شوية أو راجع الفوترة. ({detail})"
    if code == 400:
        return f"الخدمة رفضت الطلب. ({detail})"
    return f"خطأ من الخدمة ({code}): {detail}"


def complete(cfg: LLMConfig, system: str, user: str, json_mode: bool = False, max_tokens: int = 4000) -> str:
    key, model = cfg.api_key.strip(), cfg.model_name
    if not key:
        raise LLMError("مفيش مفتاح API.")
    if cfg.provider == "anthropic":
        r = _post(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": key, "anthropic-version": "2023-06-01"},
            {"model": model, "max_tokens": max_tokens, "system": system, "messages": [{"role": "user", "content": user}]},
        )
        return "".join(b.get("text", "") for b in r.get("content", []) if b.get("type") == "text").strip()
    if cfg.provider == "openai":
        body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        r = _post("https://api.openai.com/v1/chat/completions", {"Authorization": f"Bearer {key}"}, body)
        return (r["choices"][0]["message"].get("content") or "").strip()
    if cfg.provider == "gemini":
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"maxOutputTokens": max_tokens},
        }
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"
        def call(mdl: str) -> dict:
            return _post(f"https://generativelanguage.googleapis.com/v1beta/models/{mdl}:generateContent",
                         {"x-goog-api-key": key}, body)
        try:
            r = call(model)
        except LLMError as e:
            if e.suggested and not cfg.model.strip() and e.suggested != model:
                r = call(e.suggested)  # الموديل الافتراضي اتشال: نجرّب اللي اقترحته جوجل
            else:
                raise
        cands = r.get("candidates") or []
        if not cands:
            raise LLMError(f"Gemini ما رجّعش إجابة (ممكن اتحجبت). {r.get('promptFeedback', '')}")
        return "".join(p.get("text", "") for p in cands[0].get("content", {}).get("parts", [])).strip()
    raise LLMError(f"مزوّد غير معروف: {cfg.provider}")
