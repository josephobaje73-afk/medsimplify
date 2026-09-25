"""
MedSimplify - Plain-Language Drug Information App
====================================================

A Streamlit application that lets a user type a medication name and:
  1. Fetches official drug labeling info from the openFDA Drug Labeling API
  2. Checks the openFDA Recall/Enforcement API for active recalls
  3. Uses the Gemini API to rewrite dense medical text into plain language
  4. Saves every search to a local JSON file so the user can revisit history
  5. Offers live "did you mean" name suggestions while the user types,
     powered by openFDA's cheap count-query mode
  6. Lets patients submit symptom complaints to an available doctor
  7. Uses Gemini to create a non-diagnostic triage summary for each complaint
  8. Gives doctors a shared complaint inbox where they can claim and respond
  9. Shows a Profile tab with account info, usage stats, and a
     change-password form

Python concepts demonstrated (per assignment spec):
  - File handling      -> SearchHistory / ChatHistory / UserStore read and
                           write JSON files on disk
  - Exception handling -> custom exceptions for invalid names, empty results,
                           network errors, and missing fields
  - Regular expressions -> Medication.clean_text() and extract_warning_keywords()
  - OOP                -> Medication, FDAClient, AITranslator, SearchHistory,
                           ChatHistory, UserStore

Tech stack: Python, Streamlit, Requests, JSON, Gemini API, openFDA APIs.
Complaint routing and doctor availability are persisted in shared JSON files; Gemini provides the cloud-AI triage layer.

Run with:
    streamlit run app.py

Gemini cloud AI is configured in the application code.
"""

from __future__ import annotations

import hashlib
import json
import importlib
import os
import re
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Callable, ClassVar, Protocol, cast

JSONValue = dict[str, object] | list[object] | str | int | float | bool | None


class _HTTPResponse(Protocol):
    status_code: int
    ok: bool
    text: str

    def json(self) -> object:
        ...


class _RequestsExceptions(Protocol):
    RequestException: type[Exception]


class _RequestsModule(Protocol):
    exceptions: _RequestsExceptions


class _StreamlitErrorAPI(Protocol):
    def error(self, message: str) -> object:
        ...


class _StreamlitWarningAPI(Protocol):
    def warning(self, message: str) -> object:
        ...


class _StreamlitSuccessAPI(Protocol):
    def success(self, message: str) -> object:
        ...


class _StreamlitWriteAPI(Protocol):
    def write(self, message: str) -> object:
        ...


class _StreamlitExpanderAPI(Protocol):
    def __enter__(self) -> "_StreamlitExpanderAPI":
        ...

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        ...


class _StreamlitExpanderFactoryAPI(Protocol):
    def expander(self, label: str) -> _StreamlitExpanderAPI:
        ...


class _StreamlitSpinnerAPI(Protocol):
    def __enter__(self) -> "_StreamlitSpinnerAPI":
        ...

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        ...


class _StreamlitSpinnerFactoryAPI(Protocol):
    def spinner(self, text: str) -> _StreamlitSpinnerAPI:
        ...


class _StreamlitPageConfigAPI(Protocol):
    def set_page_config(
        self, *, page_title: str, page_icon: str, layout: str
    ) -> object:
        ...


class _StreamlitSidebarAPI(Protocol):
    def __enter__(self) -> "_StreamlitSidebarAPI":
        ...

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        ...

    def header(self, body: str) -> object:
        ...

    def text_input(
        self, label: str, *, value: str, type: str, help: str
    ) -> str:
        ...

    def divider(self) -> object:
        ...

    def write(self, body: str) -> object:
        ...

    def button(self, label: str) -> bool:
        ...


class _StreamlitFormAPI(Protocol):
    def __enter__(self) -> "_StreamlitFormAPI":
        ...

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        ...


class _StreamlitFormFactoryAPI(Protocol):
    def form(self, key: str, *, clear_on_submit: bool = ...) -> _StreamlitFormAPI:
        ...


class _StreamlitTabAPI(Protocol):
    def __enter__(self) -> "_StreamlitTabAPI":
        ...

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        ...


class _StreamlitTabsFactoryAPI(Protocol):
    def tabs(self, labels: list[str]) -> list[_StreamlitTabAPI]:
        ...


requests: ModuleType = importlib.import_module("requests")
RequestException = cast(_RequestsModule, cast(object, requests)).exceptions.RequestException

try:
    st: ModuleType = importlib.import_module("streamlit")
except ModuleNotFoundError:
    class _StreamlitFallback:
        def __call__(self, *args: object, **kwargs: object) -> None:
            return None

        def __getattr__(self, name: str) -> "_StreamlitFallback":
            return self

        def __enter__(self) -> "_StreamlitFallback":
            return self

        def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
            return False

    st = cast(ModuleType, cast(object, _StreamlitFallback()))


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class MedSimplifyError(Exception):
    """Base class for all app-specific errors."""


class InvalidMedicationNameError(MedSimplifyError):
    """Raised when the user's input fails basic validation."""


class MedicationNotFoundError(MedSimplifyError):
    """Raised when the openFDA label API returns no results."""


class FDANetworkError(MedSimplifyError):
    """Raised when a request to openFDA fails at the network/HTTP level."""


class AITranslationError(MedSimplifyError):
    """Raised when the Gemini API call fails or returns an unusable response."""


class ImageNotFoundError(MedSimplifyError):
    """Raised when no representative image could be found for a medication."""


class AuthenticationError(MedSimplifyError):
    """Base class for sign-up / sign-in failures."""


class UsernameTakenError(AuthenticationError):
    """Raised when registering a username that's already in use."""


class InvalidCredentialsError(AuthenticationError):
    """Raised for empty/too-short input, or a username/password that
    doesn't match a stored account. Kept deliberately generic for
    sign-in failures so we never reveal whether a username exists."""


# ---------------------------------------------------------------------------
# Medication (OOP + regex)
# ---------------------------------------------------------------------------

@dataclass
class Medication:
    """Represents a single medication's parsed label data."""

    name: str
    generic_name: str = ""
    brand_names: list[str] = field(default_factory=list)
    usage: str = ""
    warnings: str = ""
    side_effects: str = ""
    dosage: str = ""
    raw: dict[str, object] = field(default_factory=dict)

    # --- regex-based text cleaning -----------------------------------
    # FIX: these must be ClassVar, otherwise @dataclass turns them into
    # constructor parameters / instance attributes and pollutes
    # __init__, __repr__ and __eq__ with regex objects.
    _CITATION_RE: ClassVar[re.Pattern[str]] = re.compile(r"\(\d+(\.\d+)*\)")  # e.g. "(2.1)"
    _WHITESPACE_RE: ClassVar[re.Pattern[str]] = re.compile(r"\s+")
    _BULLET_RE: ClassVar[re.Pattern[str]] = re.compile(r"^\s*[\u2022\-\*]\s*", re.MULTILINE)
    _NAME_VALIDATION_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"^[A-Za-z0-9\s\-/]+$"
    )

    # A handful of clinically meaningful phrases we want to surface as
    # "keywords" to the user even before AI simplification.
    _WARNING_KEYWORD_PATTERNS: ClassVar[list[str]] = [
        r"do not use\b[^.]*",
        r"may cause\b[^.]*",
        r"risk of\b[^.]*",
        r"stop use and ask a doctor\b[^.]*",
        r"contraindicat\w*[^.]*",
        r"serious side effects?\b[^.]*",
        r"ask a doctor before use\b[^.]*",
        r"overdose\b[^.]*",
    ]

    @classmethod
    def validate_name(cls, name: str) -> str:
        """Validate and normalize a user-supplied medication name.

        Raises InvalidMedicationNameError on empty or malformed input.
        """
        cleaned = name.strip()
        if not cleaned:
            raise InvalidMedicationNameError("Medication name cannot be empty.")
        if len(cleaned) > 100:
            raise InvalidMedicationNameError("Medication name is too long.")
        if not cls._NAME_VALIDATION_RE.match(cleaned):
            raise InvalidMedicationNameError(
                "Medication name may only contain letters, numbers, spaces, hyphens, and slashes."
            )
        return cleaned

    @classmethod
    def clean_text(cls, text: str | list[str] | None) -> str:
        """Strip FDA-label boilerplate (citation markers, bullets, extra
        whitespace) from a raw label field using regular expressions."""
        if not text:
            return ""
        if isinstance(text, list):
            text = " ".join(text)
        text = cls._CITATION_RE.sub("", text)
        text = cls._BULLET_RE.sub("", text)
        text = cls._WHITESPACE_RE.sub(" ", text)
        return text.strip()

    @classmethod
    def from_openfda_result(cls, name: str, result: dict[str, object]) -> "Medication":
        """Build a Medication instance from one openFDA label API result."""
        openfda_value = result.get("openfda", {})
        openfda: dict[str, object] = {}
        if isinstance(openfda_value, dict):
            raw_openfda = cast(dict[object, object], openfda_value)
            openfda = {
                str(key): value
                for key, value in raw_openfda.items()
                if isinstance(key, str)
            }

        def text_field(key: str) -> str | list[str] | None:
            value = result.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                list_values = cast(list[object], value)
                return [item for item in list_values if isinstance(item, str)]
            return None

        def string_list_field(key: str) -> list[str]:
            value = openfda.get(key)
            if isinstance(value, list):
                list_values = cast(list[object], value)
                return [item for item in list_values if isinstance(item, str)]
            return []

        generic_names = string_list_field("generic_name")
        brand_names = string_list_field("brand_name")
        warnings = text_field("warnings") or text_field("warnings_and_cautions")

        return cls(
            name=name,
            generic_name=", ".join(generic_names) or name,
            brand_names=brand_names,
            usage=cls.clean_text(text_field("indications_and_usage")),
            warnings=cls.clean_text(warnings),
            side_effects=cls.clean_text(text_field("adverse_reactions")),
            dosage=cls.clean_text(text_field("dosage_and_administration")),
            raw=result,
        )

    def extract_warning_keywords(self) -> list[str]:
        """Use regex to pull short, human-readable warning snippets out of
        the raw warnings text (in addition to the AI-simplified version)."""
        if not self.warnings:
            return []
        found: list[str] = []
        for pattern in self._WARNING_KEYWORD_PATTERNS:
            raw_matches: list[str] = re.findall(pattern, self.warnings, flags=re.IGNORECASE)
            for match in raw_matches:
                snippet = match.strip()
                if snippet and snippet not in found:
                    found.append(snippet[:160])
        return found[:8]

    def has_missing_fields(self) -> list[str]:
        """Report which key sections came back empty from the API."""
        missing: list[str] = []
        # FIX: renamed loop variable so it doesn't shadow the `field`
        # imported from dataclasses used above in this same class body.
        for attr_name in ("usage", "warnings", "side_effects", "dosage"):
            if not getattr(self, attr_name):
                missing.append(attr_name)
        return missing


# ---------------------------------------------------------------------------
# FDAClient (OOP + exception handling)
# ---------------------------------------------------------------------------

class FDAClient:
    """Thin wrapper around the openFDA Drug Labeling and Recall/Enforcement
    (Drug Enforcement) APIs."""

    LABEL_URL: str = "https://api.fda.gov/drug/label.json"
    ENFORCEMENT_URL: str = "https://api.fda.gov/drug/enforcement.json"

    def __init__(self, timeout: int = 10):
        self.timeout: int = timeout

    def _get(self, url: str, params: dict[str, str]) -> dict[str, object]:
        try:
            response = cast(
                Callable[..., _HTTPResponse],
                requests.get,
            )(url, params=params, timeout=self.timeout)
        except RequestException as exc:
            raise FDANetworkError(f"Could not reach openFDA: {exc}") from exc

        if response.status_code == 404:
            return {"results": []}
        if not response.ok:
            raise FDANetworkError(
                f"openFDA returned an error (status {response.status_code})."
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise FDANetworkError("openFDA returned an unreadable response.") from exc

        if not isinstance(data, dict):
            raise FDANetworkError("openFDA returned an unreadable response.")
        return cast(dict[str, object], data)

    def fetch_label(self, drug_name: str) -> Medication:
        """Look up a medication by generic or brand name. Tries generic
        name first, falls back to brand name, then a free-text search."""
        queries = [
            f'openfda.generic_name:"{drug_name}"',
            f'openfda.brand_name:"{drug_name}"',
            f'indications_and_usage:"{drug_name}"',
        ]
        for query in queries:
            data = self._get(self.LABEL_URL, {"search": query, "limit": "1"})
            results_value = data.get("results")
            results_list: list[object] = (
                cast(list[object], results_value) if isinstance(results_value, list) else []
            )
            for candidate in results_list:
                if isinstance(candidate, dict):
                    return Medication.from_openfda_result(drug_name, cast(dict[str, object], candidate))

        raise MedicationNotFoundError(
            "No FDA label information found for "
            + f'"{drug_name}". Check the spelling or try the generic name.'
        )

    def suggest_names(self, partial_name: str, limit: int = 8) -> list[str]:
        """Return up to `limit` candidate generic/brand names starting with
        the given partial text, for a "did you mean" search-as-you-type
        experience. Uses openFDA's count-query mode, which returns matching
        term/count pairs instead of full label records — much cheaper than
        running a full fetch_label for every keystroke."""
        cleaned = partial_name.strip()
        if len(cleaned) < 2:
            return []

        suggestions: list[str] = []
        seen_lower: set[str] = set()

        for count_field in ("openfda.generic_name.exact", "openfda.brand_name.exact"):
            query = f"{count_field}:{cleaned}*"
            try:
                data = self._get(
                    self.LABEL_URL,
                    {"search": query, "count": count_field, "limit": str(limit)},
                )
            except FDANetworkError:
                # Suggestions are a nice-to-have, not core functionality —
                # a transient failure here shouldn't block the user from
                # typing a name and searching normally.
                continue

            results_value = data.get("results")
            if not isinstance(results_value, list):
                continue

            for item in cast(list[object], results_value):
                if not isinstance(item, dict):
                    continue
                item_dict = cast(dict[str, object], item)
                term = item_dict.get("term")
                if not isinstance(term, str):
                    continue
                normalized = term.strip().title()
                if normalized and normalized.lower() not in seen_lower:
                    seen_lower.add(normalized.lower())
                    suggestions.append(normalized)

        return suggestions[:limit]

    def fetch_recalls(self, drug_name: str, limit: int = 5) -> list[dict[str, str]]:
        """Return recent recall/enforcement records mentioning this drug."""
        query = f'product_description:"{drug_name}"'
        data = self._get(
            self.ENFORCEMENT_URL,
            {"search": query, "limit": str(limit), "sort": "report_date:desc"},
        )
        results_value = data.get("results")
        if not isinstance(results_value, list):
            return []

        recalls: list[dict[str, str]] = []
        results_list: list[object] = cast(list[object], results_value)
        for item in results_list:
            if not isinstance(item, dict):
                continue
            item_dict = cast(dict[str, object], item)
            product_description = item_dict.get("product_description")
            reason_for_recall = item_dict.get("reason_for_recall")
            classification = item_dict.get("classification", "Unknown")
            status = item_dict.get("status", "Unknown")
            report_date = item_dict.get("report_date", "")
            recalling_firm = item_dict.get("recalling_firm", "")

            product_description_text = ""
            if isinstance(product_description, str):
                product_description_text = product_description
            elif isinstance(product_description, list):
                product_description_parts = cast(list[object], product_description)
                product_description_text = " ".join(
                    str(part) for part in product_description_parts if isinstance(part, str)
                )

            reason_for_recall_text = ""
            if isinstance(reason_for_recall, str):
                reason_for_recall_text = reason_for_recall
            elif isinstance(reason_for_recall, list):
                reason_for_recall_parts = cast(list[object], reason_for_recall)
                reason_for_recall_text = " ".join(
                    str(part) for part in reason_for_recall_parts if isinstance(part, str)
                )

            recalls.append(
                {
                    "product_description": Medication.clean_text(product_description_text),
                    "reason_for_recall": Medication.clean_text(reason_for_recall_text),
                    "classification": str(classification),
                    "status": str(status),
                    "report_date": str(report_date),
                    "recalling_firm": str(recalling_firm),
                }
            )
        return recalls


# ---------------------------------------------------------------------------
# AITranslator (OOP + exception handling)
# ---------------------------------------------------------------------------

class AITranslator:
    """Calls the Gemini API to rewrite medical text in plain language, and
    to power the free-form Chat tab."""

    API_URL_TEMPLATE: str = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "{model}:generateContent"
    )
    OPENAI_API_URL: str = (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )

    # NEW: system-style framing prepended to every chat conversation, sent
    # as a fake first user/model turn since the generateContent endpoint
    # used here doesn't take a separate system-instruction field in this
    # minimal integration. Keeps the assistant on-topic and reminds it not
    # to replace a real clinician.
    _CHAT_INTRO: str = (
        "You are a helpful assistant inside MedSimplify, an app that "
        "explains FDA drug label information in plain language. Answer "
        "the user's questions about medications, general health topics, "
        "or how to use the app. You are not a doctor — remind the user "
        "to consult a healthcare professional for personal medical "
        "advice when appropriate. Keep answers concise and easy to "
        "understand."
    )
    _CHAT_INTRO_ACK: str = (
        "Understood — I'll answer clearly and suggest seeing a "
        "healthcare professional for personal medical advice."
    )
    _MAX_CHAT_HISTORY_TURNS: int = 20

    def __init__(self, api_key: str, model: str = "gemini-3.8-flash"):
        if not api_key:
            raise AITranslationError("No Gemini API key was provided.")
        self.api_key: str = api_key.strip().strip("\"'")
        self.model: str = model
        self._response_mode: str = "native"

    @staticmethod
    def _to_openai_messages(contents: list[dict[str, object]]) -> list[dict[str, str]]:
        """Convert Gemini generateContent turns to OpenAI-compatible messages."""
        messages: list[dict[str, str]] = []
        for item in contents:
            role = str(item.get("role", "user"))
            if role == "model":
                role = "assistant"
            parts = item.get("parts", [])
            text_value = ""
            if isinstance(parts, list) and parts:
                first = parts[0]
                if isinstance(first, dict):
                    candidate = first.get("text")
                    if isinstance(candidate, str):
                        text_value = candidate
            if text_value:
                messages.append({"role": role, "content": text_value})
        return messages

    def _post(self, contents: list[dict[str, object]]) -> _HTTPResponse:
        """Call Gemini using native API-key auth, with a compatibility fallback.

        Native generateContent uses x-goog-api-key. Some current AQ-format
        authorization keys can return 401 on the native gateway; the official
        OpenAI-compatible Gemini endpoint accepts the same key as a Bearer
        token, so retry there only when native authentication is rejected.
        """
        native_url = self.API_URL_TEMPLATE.format(model=self.model)
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        native_payload = {"contents": contents}

        try:
            response: _HTTPResponse = cast(Callable[..., _HTTPResponse], requests.post)(
                native_url,
                headers=headers,
                data=json.dumps(native_payload),
                timeout=20,
            )
        except RequestException as exc:
            raise AITranslationError(f"Could not reach Gemini API: {exc}") from exc

        if response.ok:
            self._response_mode = "native"
            return response

        if response.status_code != 401:
            raise AITranslationError(
                f"Gemini API returned status {response.status_code}: {response.text[:300]}"
            )

        # Fallback for current AQ/auth-key gateway compatibility issues.
        openai_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        openai_payload = {
            "model": self.model,
            "messages": self._to_openai_messages(contents),
        }
        try:
            fallback: _HTTPResponse = cast(Callable[..., _HTTPResponse], requests.post)(
                self.OPENAI_API_URL,
                headers=openai_headers,
                data=json.dumps(openai_payload),
                timeout=20,
            )
        except RequestException as exc:
            raise AITranslationError(f"Could not reach Gemini compatibility API: {exc}") from exc

        if not fallback.ok:
            raise AITranslationError(
                "Gemini authentication failed on both API endpoints. "
                f"Native: {response.status_code}; compatibility: "
                f"{fallback.status_code}: {fallback.text[:250]}"
            )

        self._response_mode = "openai"
        return fallback

    @staticmethod
    def _extract_text(response: _HTTPResponse) -> str:
        """Pull the first candidate's text out of a generateContent
        response. Shared by simplify() and chat() so the (fairly deep)
        response-shape validation only lives in one place.
        """
        try:
            response_payload = response.json()
        except ValueError as exc:
            raise AITranslationError("Gemini API returned an unexpected response.") from exc

        if not isinstance(response_payload, dict):
            raise AITranslationError("Gemini API returned an unexpected response.")

        payload_dict = cast(dict[str, object], response_payload)

        if self._response_mode == "openai":
            choices = payload_dict.get("choices")
            if not isinstance(choices, list) or not choices:
                raise AITranslationError("Gemini compatibility API returned an unexpected response.")
            first_choice = choices[0]
            if not isinstance(first_choice, dict):
                raise AITranslationError("Gemini compatibility API returned an unexpected response.")
            message = first_choice.get("message")
            if not isinstance(message, dict):
                raise AITranslationError("Gemini compatibility API returned an unexpected response.")
            text_value = message.get("content")
            if not isinstance(text_value, str):
                raise AITranslationError("Gemini compatibility API returned an unexpected response.")
            return text_value.strip()

        candidates = payload_dict.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise AITranslationError("Gemini API returned an unexpected response.")

        first_candidate = cast(list[object], candidates)[0]
        if not isinstance(first_candidate, dict):
            raise AITranslationError("Gemini API returned an unexpected response.")

        candidate_dict = cast(dict[str, object], first_candidate)
        content = candidate_dict.get("content")
        if not isinstance(content, dict):
            raise AITranslationError("Gemini API returned an unexpected response.")

        content_dict = cast(dict[str, object], content)
        parts = content_dict.get("parts")
        if not isinstance(parts, list) or not parts:
            raise AITranslationError("Gemini API returned an unexpected response.")

        first_part = cast(list[object], parts)[0]
        if not isinstance(first_part, dict):
            raise AITranslationError("Gemini API returned an unexpected response.")

        part_dict = cast(dict[str, object], first_part)
        text_value = part_dict.get("text")
        if not isinstance(text_value, str):
            raise AITranslationError("Gemini API returned an unexpected response.")

        return text_value.strip()

    def simplify(self, text: str, section_name: str) -> str:
        """Rewrite one section of drug info in everyday language."""
        if not text:
            return "No information was provided by the FDA for this section."

        prompt = (
            "You are helping a patient with no medical background understand "
            f"their medication. Rewrite the following '{section_name}' section "
            "from an official FDA drug label in simple, clear, everyday "
            "language. Keep it accurate, use short sentences, and avoid "
            "jargon. Limit your answer to about 120 words.\n\n"
            f"Original text:\n{text}"
        )

        contents: list[dict[str, object]] = [{"role": "user", "parts": [{"text": prompt}]}]
        response = self._post(contents)
        return self._extract_text(response)

    def chat(self, message: str, history: list[dict[str, str]]) -> str:
        """Have one free-form conversation turn with Gemini.

        `history` is the prior turns as a list of {"role": "user"|"model",
        "content": str} dicts (oldest first, NOT including `message`
        itself). Returns the assistant's reply text.
        """
        cleaned_message = message.strip()
        if not cleaned_message:
            raise AITranslationError("Message cannot be empty.")

        contents: list[dict[str, object]] = [
            {"role": "user", "parts": [{"text": self._CHAT_INTRO}]},
            {"role": "model", "parts": [{"text": self._CHAT_INTRO_ACK}]},
        ]

        # Keep the payload bounded — only send the most recent turns.
        for turn in history[-self._MAX_CHAT_HISTORY_TURNS:]:
            role = turn.get("role", "user")
            turn_content = turn.get("content", "")
            if role not in ("user", "model") or not turn_content:
                continue
            contents.append({"role": role, "parts": [{"text": turn_content}]})

        contents.append({"role": "user", "parts": [{"text": cleaned_message}]})

        response = self._post(contents)
        return self._extract_text(response)


# ---------------------------------------------------------------------------
# ImageClient (OOP + exception handling)
# ---------------------------------------------------------------------------

class ImageClient:
    """Fetches a representative thumbnail image for a medication from
    Wikipedia's public REST API (no API key required)."""

    SUMMARY_URL_TEMPLATE: str = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}" 
    # Wikipedia's API etiquette policy rejects requests with no descriptive 
    # User-Agent (typically with a 403), so this must be sent on every call. 
    USER_AGENT: str = "MedSimplify/1.0 (educational project; contact: example@example.com)" 
 
    def __init__(self, timeout: int = 10): 
        self.timeout: int = timeout 
 
    def _fetch_thumbnail_for_title(self, title: str) -> str | None: 
        """Try one exact page title. Returns the thumbnail URL, or None if 
        that specific title has no page / no image (not an error yet — 
        the caller may have other names left to try).""" 
        url = self.SUMMARY_URL_TEMPLATE.format(title=title.strip().replace(" ", "_")) 
 
        try: 
            response = cast( 
                Callable[..., _HTTPResponse], 
                requests.get, 
            )(url, headers={"User-Agent": self.USER_AGENT}, timeout=self.timeout) 
        except RequestException as exc: 
            raise ImageNotFoundError(f"Could not reach Wikipedia: {exc}") from exc 
 
        if response.status_code == 404: 
            return None 
        if not response.ok: 
            raise ImageNotFoundError( 
                f"Wikipedia returned an error (status {response.status_code})." 
            ) 
 
        try: 
            data = response.json() 
        except ValueError as exc: 
            raise ImageNotFoundError("Wikipedia returned an unreadable response.") from exc 
 
        if not isinstance(data, dict): 
            return None 
 
        data_dict = cast(dict[str, object], data) 
        thumbnail = data_dict.get("thumbnail") 
        if not isinstance(thumbnail, dict): 
            return None 
 
        thumbnail_dict = cast(dict[str, object], thumbnail) 
        source = thumbnail_dict.get("source") 
        if not isinstance(source, str): 
            return None 
 
        return source 
 
    def fetch_thumbnail_url(self, medication: "Medication", fallback_name: str) -> str: 
        """Try the medication's generic name first, then each brand name, 
        then the raw search term the user typed. Returns the first 
        thumbnail found; raises ImageNotFoundError if none of them have one.""" 
        candidates: list[str] = [] 
        if medication.generic_name: 
            candidates.append(medication.generic_name) 
        candidates.extend(medication.brand_names) 
        if fallback_name not in candidates: 
            candidates.append(fallback_name) 
 
        for candidate_name in candidates: 
            thumbnail_url = self._fetch_thumbnail_for_title(candidate_name) 
            if thumbnail_url: 
                return thumbnail_url 
 
        raise ImageNotFoundError( 
            f'No image available for "{fallback_name}" (tried: {", ".join(candidates)}).' 
        ) 
 
 
# --------------------------------------------------------------------------- 
# User accounts (OOP + file handling + password security) 
# --------------------------------------------------------------------------- 
 
@dataclass 
class User: 
    """A registered user account (never holds a plain-text password).""" 
 
    username: str 
    password_hash: str 
    salt: str 
    created_at: str 
    role: str = "user"
    full_name: str = ""
    specialty: str = ""
    license_number: str = ""
    certificate_name: str = ""
 
 
class UserStore: 
    """Persists user accounts to a local JSON file. Passwords are never 
    stored in plain text: each one is salted with a random per-user value 
    and stretched through PBKDF2-HMAC-SHA256 before being written to disk, 
    and login compares hashes using a constant-time comparison to avoid 
    leaking timing information about how close a guess was.""" 
 
    _HASH_ITERATIONS: ClassVar[int] = 200_000 
    _MIN_PASSWORD_LENGTH: ClassVar[int] = 8 
 
    def __init__(self, filepath: str = "users.json"): 
        self.filepath: Path = Path(filepath) 
        if not self.filepath.exists(): 
            self._write({}) 
 
    def _read(self) -> dict[str, dict[str, object]]: 
        try: 
            with self.filepath.open("r", encoding="utf-8") as f: 
                loaded = cast(object, json.load(f)) 
        except (json.JSONDecodeError, FileNotFoundError): 
            return {} 
 
        if not isinstance(loaded, dict): 
            return {} 
 
        users: dict[str, dict[str, object]] = {} 
        for key, value in cast(dict[object, object], loaded).items(): 
            if isinstance(key, str) and isinstance(value, dict): 
                users[key] = cast(dict[str, object], value) 
        return users 
 
    def _write(self, users: dict[str, dict[str, object]]) -> None: 
        with self.filepath.open("w", encoding="utf-8") as f: 
            json.dump(users, f, indent=2, ensure_ascii=False) 
 
    @staticmethod 
    def _hash_password(password: str, salt_hex: str) -> str: 
        """PBKDF2-HMAC-SHA256 the password against the given hex-encoded 
        salt. Deterministic for a given (password, salt) pair, which is 
        what lets sign-in re-derive and compare against the stored hash.""" 
        digest = hashlib.pbkdf2_hmac( 
            "sha256", 
            password.encode("utf-8"), 
            bytes.fromhex(salt_hex), 
            UserStore._HASH_ITERATIONS, 
        ) 
        return digest.hex() 
 
    def register(self, username: str, password: str, confirm_password: str) -> User: 
        """Create a new account. Raises InvalidCredentialsError on bad 
        input, or UsernameTakenError if the username is already in use.""" 
        cleaned_username = username.strip() 
        if not cleaned_username: 
            raise InvalidCredentialsError("Username cannot be empty.") 
        if len(cleaned_username) > 50: 
            raise InvalidCredentialsError("Username is too long.") 
        if len(password) < self._MIN_PASSWORD_LENGTH: 
            raise InvalidCredentialsError( 
                f"Password must be at least {self._MIN_PASSWORD_LENGTH} characters long." 
            ) 
        if password != confirm_password: 
            raise InvalidCredentialsError("Passwords do not match.") 
 
        users = self._read() 
        lookup_key = cleaned_username.lower() 
        if lookup_key in users: 
            raise UsernameTakenError(f'The username "{cleaned_username}" is already taken.') 
 
        salt_hex = secrets.token_hex(16) 
        password_hash = self._hash_password(password, salt_hex) 
        created_at = datetime.now().isoformat(timespec="seconds") 
 
        users[lookup_key] = { 
            "username": cleaned_username, 
            "password_hash": password_hash, 
            "salt": salt_hex, 
            "created_at": created_at, 
        } 
        self._write(users) 
 
        return User( 
            username=cleaned_username, 
            password_hash=password_hash, 
            salt=salt_hex, 
            created_at=created_at, 
        ) 
 
    def register_doctor(
        self, username: str, password: str, confirm_password: str,
        full_name: str, specialty: str, license_number: str, certificate_name: str,
    ) -> User:
        """Create a doctor account with professional credential details."""
        cleaned_name = full_name.strip()
        cleaned_specialty = specialty.strip()
        cleaned_license = license_number.strip()
        cleaned_certificate = certificate_name.strip()
        if not cleaned_name:
            raise InvalidCredentialsError("Full professional name is required.")
        if not cleaned_specialty:
            raise InvalidCredentialsError("Medical specialty is required.")
        if not cleaned_license:
            raise InvalidCredentialsError("Medical license or certificate number is required.")
        if not cleaned_certificate:
            raise InvalidCredentialsError("A medical certificate must be uploaded.")
        user = self.register(username, password, confirm_password)
        users = self._read()
        key = user.username.lower()
        users[key].update({
            "role": "doctor", "full_name": cleaned_name,
            "specialty": cleaned_specialty, "license_number": cleaned_license,
            "certificate_name": cleaned_certificate,
        })
        self._write(users)
        return User(user.username, user.password_hash, user.salt, user.created_at,
                    "doctor", cleaned_name, cleaned_specialty, cleaned_license, cleaned_certificate)

    def authenticate(self, username: str, password: str) -> User: 
        """Verify credentials against a stored account. Raises 
        InvalidCredentialsError on any mismatch — deliberately the same 
        error whether the username doesn't exist or the password is 
        wrong, so a failed attempt never reveals which one was incorrect.""" 
        users = self._read() 
        record = users.get(username.strip().lower()) 
        if record is None: 
            raise InvalidCredentialsError("Incorrect username or password.") 
 
        stored_salt = record.get("salt") 
        stored_hash = record.get("password_hash") 
        if not isinstance(stored_salt, str) or not isinstance(stored_hash, str): 
            raise InvalidCredentialsError("Incorrect username or password.") 
 
        attempted_hash = self._hash_password(password, stored_salt) 
        if not secrets.compare_digest(attempted_hash, stored_hash): 
            raise InvalidCredentialsError("Incorrect username or password.") 
 
        return User( 
            username=str(record.get("username", username)), 
            password_hash=stored_hash, 
            salt=stored_salt, 
            created_at=str(record.get("created_at", "")),
            role=str(record.get("role", "user")),
            full_name=str(record.get("full_name", "")),
            specialty=str(record.get("specialty", "")),
            license_number=str(record.get("license_number", "")),
            certificate_name=str(record.get("certificate_name", "")),
        ) 
 
    def get_user(self, username: str) -> User | None: 
        """Look up a stored account by username without a password check. 
        Used to populate the Profile tab for the already-authenticated 
        current user. Returns None if the account no longer exists.""" 
        users = self._read() 
        record = users.get(username.strip().lower()) 
        if record is None: 
            return None 
 
        stored_salt = record.get("salt") 
        stored_hash = record.get("password_hash") 
        if not isinstance(stored_salt, str) or not isinstance(stored_hash, str): 
            return None 
 
        return User( 
            username=str(record.get("username", username)), 
            password_hash=stored_hash, 
            salt=stored_salt, 
            created_at=str(record.get("created_at", "")),
            role=str(record.get("role", "user")),
            full_name=str(record.get("full_name", "")),
            specialty=str(record.get("specialty", "")),
            license_number=str(record.get("license_number", "")),
            certificate_name=str(record.get("certificate_name", "")),
        ) 
 
    def change_password( 
        self, 
        username: str, 
        current_password: str, 
        new_password: str, 
        confirm_new_password: str, 
    ) -> None: 
        """Verify current_password against the stored account, then 
        replace it with a freshly salted/hashed new_password. Raises 
        InvalidCredentialsError on any validation failure (including a 
        wrong current password).""" 
        # Re-uses authenticate()'s constant-time comparison and its 
        # deliberately generic error message. 
        self.authenticate(username, current_password) 
 
        if len(new_password) < self._MIN_PASSWORD_LENGTH: 
            raise InvalidCredentialsError( 
                f"New password must be at least {self._MIN_PASSWORD_LENGTH} characters long." 
            ) 
        if new_password != confirm_new_password: 
            raise InvalidCredentialsError("New passwords do not match.") 
 
        users = self._read() 
        lookup_key = username.strip().lower() 
        record = users.get(lookup_key) 
        if record is None: 
            raise InvalidCredentialsError("Incorrect username or password.") 
 
        salt_hex = secrets.token_hex(16) 
        record["password_hash"] = self._hash_password(new_password, salt_hex) 
        record["salt"] = salt_hex 
        users[lookup_key] = record 
        self._write(users) 
 
 
# --------------------------------------------------------------------------- 
# SearchHistory (OOP + file handling) 
# --------------------------------------------------------------------------- 
 
# ---------------------------------------------------------------------------
# Complaint queue + doctor availability
# ---------------------------------------------------------------------------

class ComplaintStore:
    """Persist patient complaints and doctor responses in shared JSON storage."""
    def __init__(self, filepath: str = "complaints.json") -> None:
        self.filepath = Path(filepath)
        if not self.filepath.exists(): self._write([])
    def _read(self) -> list[dict[str, object]]:
        try:
            with self.filepath.open("r", encoding="utf-8") as f: loaded = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError): return []
        return [dict(x) for x in loaded] if isinstance(loaded, list) else []
    def _write(self, complaints: list[dict[str, object]]) -> None:
        with self.filepath.open("w", encoding="utf-8") as f: json.dump(complaints, f, indent=2, ensure_ascii=False)
    def create(self, patient_username: str, complaint: str, triage: str) -> dict[str, object]:
        complaints=self._read(); now=datetime.now().isoformat(timespec="seconds")
        record={"id":secrets.token_urlsafe(12),"patient_username":patient_username,"complaint":complaint,"triage":triage,"status":"waiting","assigned_doctor":"","doctor_response":"","created_at":now,"updated_at":now}
        complaints.append(record); self._write(complaints); return record
    def get_waiting(self) -> list[dict[str, object]]:
        return [c for c in self._read() if str(c.get("status","waiting")) == "waiting"]
    def get_for_patient(self, username: str) -> list[dict[str, object]]:
        return [c for c in self._read() if str(c.get("patient_username","")).lower()==username.lower()]
    def get_for_doctor(self, username: str) -> list[dict[str, object]]:
        return [c for c in self._read() if str(c.get("assigned_doctor","")).lower()==username.lower()]
    def claim(self, complaint_id: str, doctor_username: str) -> bool:
        complaints=self._read()
        for c in complaints:
            if str(c.get("id",""))==complaint_id:
                if str(c.get("status","waiting"))!="waiting": return False
                c["assigned_doctor"]=doctor_username; c["status"]="claimed"; c["updated_at"]=datetime.now().isoformat(timespec="seconds"); self._write(complaints); return True
        return False
    def respond(self, complaint_id: str, doctor_username: str, response: str) -> bool:
        if not response.strip(): return False
        complaints=self._read()
        for c in complaints:
            if str(c.get("id",""))==complaint_id and str(c.get("assigned_doctor","")).lower()==doctor_username.lower():
                c["doctor_response"]=response.strip(); c["status"]="responded"; c["updated_at"]=datetime.now().isoformat(timespec="seconds"); self._write(complaints); return True
        return False

class DoctorAvailabilityStore:
    """Persist which doctors are currently marked online."""
    def __init__(self, filepath: str = "doctor_availability.json") -> None:
        self.filepath=Path(filepath)
        if not self.filepath.exists(): self._write({})
    def _read(self) -> dict[str, dict[str, object]]:
        try:
            with self.filepath.open("r", encoding="utf-8") as f: loaded=json.load(f)
        except (json.JSONDecodeError, FileNotFoundError): return {}
        return {str(k):dict(v) for k,v in loaded.items() if isinstance(v,dict)} if isinstance(loaded,dict) else {}
    def _write(self, doctors: dict[str, dict[str, object]]) -> None:
        with self.filepath.open("w", encoding="utf-8") as f: json.dump(doctors,f,indent=2,ensure_ascii=False)
    def set_online(self, username: str, license_number: str, certificate_name: str) -> None:
        d=self._read(); d[username.lower()]={"username":username,"license_number":license_number,"certificate_name":certificate_name,"online":True,"updated_at":datetime.now().isoformat(timespec="seconds")}; self._write(d)
    def set_offline(self, username: str) -> None:
        d=self._read(); key=username.lower()
        if key in d: d[key]["online"]=False; d[key]["updated_at"]=datetime.now().isoformat(timespec="seconds"); self._write(d)
    def available_doctors(self) -> list[dict[str, object]]:
        return [x for x in self._read().values() if bool(x.get("online",False))]
    def assign_available_doctor(self, complaint_id: str, complaints: ComplaintStore) -> str | None:
        available=self.available_doctors()
        if not available: return None
        selected=min(available,key=lambda d:len(complaints.get_for_doctor(str(d.get("username","")))))
        username=str(selected.get("username",""))
        return username if complaints.claim(complaint_id,username) else None


class SearchHistory: 
    """Persists search results to a local JSON file.""" 
 
    def __init__(self, filepath: str = "search_history.json"): 
        self.filepath: Path = Path(filepath) 
        if not self.filepath.exists(): 
            self._write([]) 
 
    def _read(self) -> list[dict[str, object]]: 
        try: 
            with self.filepath.open("r", encoding="utf-8") as f: 
                loaded = cast(object, json.load(f)) 
        except (json.JSONDecodeError, FileNotFoundError): 
            return [] 
 
        if not isinstance(loaded, list): 
            return [] 
 
        entries: list[dict[str, object]] = [] 
        for entry in cast(list[object], loaded): 
            if isinstance(entry, dict): 
                entries.append(cast(dict[str, object], entry)) 
        return entries 
 
    def _write(self, entries: list[dict[str, object]]) -> None: 
        with self.filepath.open("w", encoding="utf-8") as f: 
            json.dump(entries, f, indent=2, ensure_ascii=False) 
 
    def add(self, entry: dict[str, object]) -> None: 
        entries = self._read() 
        entries.insert(0, entry)  # newest first 
        self._write(entries[:100])  # cap history size 
 
    def get_all(self) -> list[dict[str, object]]: 
        return self._read() 
 
    def clear(self) -> None: 
        self._write([]) 
 
 
# --------------------------------------------------------------------------- 
# ChatHistory (OOP + file handling) 
# --------------------------------------------------------------------------- 
 
class ChatHistory: 
    """Persists one user's AI-chat conversation to a local JSON file, so 
    the conversation survives across Streamlit reruns (every widget 
    interaction reruns the whole script) and app restarts.""" 
 
    def __init__(self, filepath: str = "chat_history.json"): 
        self.filepath: Path = Path(filepath) 
        if not self.filepath.exists(): 
            self._write([]) 
 
    def _read(self) -> list[dict[str, str]]: 
        try: 
            with self.filepath.open("r", encoding="utf-8") as f: 
                loaded = cast(object, json.load(f)) 
        except (json.JSONDecodeError, FileNotFoundError): 
            return [] 
 
        if not isinstance(loaded, list): 
            return [] 
 
        messages: list[dict[str, str]] = [] 
        for entry in cast(list[object], loaded): 
            if not isinstance(entry, dict): 
                continue 
            entry_dict = cast(dict[str, object], entry) 
            role = entry_dict.get("role") 
            content = entry_dict.get("content") 
            if isinstance(role, str) and isinstance(content, str): 
                messages.append({"role": role, "content": content}) 
        return messages 
 
    def _write(self, messages: list[dict[str, str]]) -> None: 
        with self.filepath.open("w", encoding="utf-8") as f: 
            json.dump(messages, f, indent=2, ensure_ascii=False) 
 
    def add(self, role: str, content: str) -> None: 
        messages = self._read() 
        messages.append({"role": role, "content": content}) 
        self._write(messages[-200:])  # cap history size 
 
    def get_all(self) -> list[dict[str, str]]: 
        return self._read() 
 
    def clear(self) -> None: 
        self._write([]) 
 
 
# --------------------------------------------------------------------------- 
# Profile stats (plain function — no state of its own) 
# --------------------------------------------------------------------------- 
 
def compute_profile_stats(entries: list[dict[str, object]]) -> dict[str, int]: 
    """Summarize a user's search history for display on the Profile tab.""" 
    total_searches = len(entries) 
    recalls_flagged = sum(1 for e in entries if e.get("recall_found")) 
    return { 
        "total_searches": total_searches, 
        "recalls_flagged": recalls_flagged, 
    } 
 
 
# --------------------------------------------------------------------------- 
# Streamlit UI 
# --------------------------------------------------------------------------- 
 
@lru_cache(maxsize=256) 
def get_cached_suggestions(partial_name: str) -> tuple[str, ...]: 
    """Look up "did you mean" name suggestions for a partial medication 
    name, cached per unique prefix. Streamlit reruns the whole script on 
    every widget interaction, so without this cache the same prefix (e.g. 
    while the user is deciding whether to keep typing) would re-hit the 
    openFDA count endpoint on every rerun. Suggestions are best-effort: 
    any network problem here is swallowed rather than shown to the user, 
    since a failed suggestion lookup should never block a normal search.""" 
    try: 
        return tuple(FDAClient().suggest_names(partial_name)) 
    except FDANetworkError: 
        return () 
 
 
def build_entry( 
    drug_name: str, 
    medication: Medication, 
    recalls: list[dict[str, str]], 
    simplified: dict[str, str], 
) -> dict[str, object]: 
    return { 
        "timestamp": datetime.now().isoformat(timespec="seconds"), 
        "query": drug_name, 
        "generic_name": medication.generic_name, 
        "brand_names": medication.brand_names, 
        "recall_found": bool(recalls), 
        "recall_count": len(recalls), 
        "simplified": simplified, 
    } 
 
 
def render_recall_banner(recalls: list[dict[str, str]]) -> None: 
    if recalls: 
        _ = cast(_StreamlitErrorAPI, cast(object, st)).error( 
            f"⚠️ {len(recalls)} recall notice(s) found for this medication." 
        ) 
        for r in recalls: 
            title = ( 
                f"{r['recalling_firm'] or 'Unknown firm'} — {r['report_date']} " 
                + f"(Class {r['classification']})" 
            ) 
            with cast(_StreamlitExpanderFactoryAPI, cast(object, st)).expander(title): 
                _ = cast(_StreamlitWriteAPI, cast(object, st)).write( 
                    f"**Reason:** {r['reason_for_recall'] or 'Not specified'}" 
                ) 
                _ = cast(_StreamlitWriteAPI, cast(object, st)).write( 
                    f"**Status:** {r['status']}" 
                ) 
                _ = cast(_StreamlitWriteAPI, cast(object, st)).write( 
                    f"**Product:** {r['product_description']}" 
                ) 
    else: 
        _ = cast(_StreamlitSuccessAPI, cast(object, st)).success( 
            "✅ No recent recalls found for this medication." 
        ) 
 
 
_SUGGESTION_PLACEHOLDER: str = "Did you mean…" 
 
 
def render_search_suggestions(session_state: dict[str, object], drug_name_input: str) -> None: 
    """Render a "did you mean" dropdown under the search box, driven by 
    FDAClient.suggest_names(). Picking an option fills the search box 
    with the exact FDA-recognized name and immediately triggers a 
    search, so the user doesn't have to click Search a second time. 
 
    The dropdown's widget key is derived from the current prefix 
    (`suggestion_select::<prefix>`) rather than a single fixed key. 
    Streamlit requires a selectbox's stored value to be one of its 
    current `options`, and the suggestion list changes on every 
    keystroke — a fixed key would raise once a previously-selected 
    option disappeared from a new options list. Keying per prefix 
    sidesteps that: each prefix gets its own fresh widget state, so 
    there's never a stale selection sitting outside the current options. 
    """ 
    cleaned = drug_name_input.strip() 
    if len(cleaned) < 2: 
        return 
 
    suggestions = get_cached_suggestions(cleaned) 
    # Don't show a suggestion that's just an exact echo of what's 
    # already typed — that's not a useful "did you mean". 
    suggestions = tuple(s for s in suggestions if s.lower() != cleaned.lower()) 
    if not suggestions: 
        return 
 
    dropdown_key = f"suggestion_select::{cleaned.lower()}" 
    options = [_SUGGESTION_PLACEHOLDER, *suggestions] 
 
    def _apply_selected_suggestion() -> None: 
        selected = cast(str, session_state.get(dropdown_key, _SUGGESTION_PLACEHOLDER)) 
        if selected and selected != _SUGGESTION_PLACEHOLDER: 
            # Setting these here is what lets the text_input (bound to 
            # the same session_state key) pick up the chosen name on 
            # the rerun Streamlit already triggers after on_change, and 
            # auto-search it without a second click. 
            session_state["drug_name_input"] = selected 
            session_state["trigger_search"] = True 
 
    cast(Callable[..., object], getattr(cast(object, st), "selectbox"))( 
        "Did you mean:", 
        options, 
        key=dropdown_key, 
        on_change=_apply_selected_suggestion, 
    ) 
 
 
def render_auth_ui(user_store: UserStore, session_state: dict[str, object]) -> None: 
    """Render the sign-in / doctor sign-in / sign-up screen. Doctor access 
    requires normal account credentials plus a medical certificate/license 
    number and an uploaded certificate before the app can be accessed.""" 
    cast(Callable[[str], object], getattr(cast(object, st), "subheader"))( 
        "Sign in to continue" 
    ) 
    cast(Callable[[str], object], getattr(cast(object, st), "caption"))( 
        "Doctors can use the Doctor Sign In section to access the shared complaint " 
        + "inbox after providing their professional certificate details." 
    ) 
 
    tabs = cast(_StreamlitTabsFactoryAPI, cast(object, st)).tabs( 
        ["Sign In", "Doctor Sign In", "Patient Sign Up", "Doctor Sign Up"] 
    ) 
 
    # --- Sign In --------------------------------------------------------- 
    with tabs[0]: 
        with cast(_StreamlitFormFactoryAPI, cast(object, st)).form("sign_in_form"): 
            sign_in_username = cast( 
                Callable[..., str], getattr(cast(object, st), "text_input") 
            )("Username", key="sign_in_username") 
            sign_in_password = cast( 
                Callable[..., str], getattr(cast(object, st), "text_input") 
            )("Password", type="password", key="sign_in_password") 
            sign_in_submitted = cast( 
                Callable[..., bool], getattr(cast(object, st), "form_submit_button") 
            )("Sign In") 
 
        if sign_in_submitted: 
            try: 
                user = user_store.authenticate(sign_in_username, sign_in_password) 
            except InvalidCredentialsError as exc: 
                _ = cast(_StreamlitErrorAPI, cast(object, st)).error(str(exc)) 
            else: 
                session_state["logged_in_user"] = user.username 
                session_state["user_role"] = "user" 
                session_state["doctor_certificate_name"] = "" 
                session_state["doctor_license_number"] = "" 
                cast(Callable[[], object], getattr(cast(object, st), "rerun"))() 
 
    # --- Doctor Sign In -------------------------------------------------- 
    with tabs[1]: 
        cast(Callable[[str], object], getattr(cast(object, st), "info"))( 
            "Doctor access is intended for licensed medical professionals. " 
            + "Certificate upload is a credential-submission step; this local " 
            + "app does not independently verify the certificate with a " 
            + "medical licensing authority." 
        ) 
        with cast(_StreamlitFormFactoryAPI, cast(object, st)).form( 
            "doctor_sign_in_form" 
        ): 
            doctor_username = cast( 
                Callable[..., str], getattr(cast(object, st), "text_input") 
            )("Username", key="doctor_sign_in_username") 
            doctor_password = cast( 
                Callable[..., str], getattr(cast(object, st), "text_input") 
            )("Password", type="password", key="doctor_sign_in_password") 
            doctor_license_number = cast( 
                Callable[..., str], getattr(cast(object, st), "text_input") 
            )("Medical license / certificate number", key="doctor_sign_in_license_number") 
            doctor_certificate = cast( 
                Callable[..., object], getattr(cast(object, st), "file_uploader") 
            )( 
                "Upload medical certificate", 
                type=["pdf", "png", "jpg", "jpeg"], 
                key="doctor_certificate", 
                help="Upload the professional certificate that supports your doctor access request.", 
            ) 
            doctor_submitted = cast( 
                Callable[..., bool], getattr(cast(object, st), "form_submit_button") 
            )("Doctor Sign In") 
 
        if doctor_submitted: 
            if not doctor_license_number.strip(): 
                _ = cast(_StreamlitErrorAPI, cast(object, st)).error( 
                    "Enter your medical license or certificate number." 
                ) 
            elif doctor_certificate is None: 
                _ = cast(_StreamlitErrorAPI, cast(object, st)).error( 
                    "Upload your medical certificate before accessing the app." 
                ) 
            else: 
                try: 
                    user = user_store.authenticate(doctor_username, doctor_password) 
                except InvalidCredentialsError as exc: 
                    _ = cast(_StreamlitErrorAPI, cast(object, st)).error(str(exc)) 
                else:
                    if user.role != "doctor":
                        _ = cast(_StreamlitErrorAPI, cast(object, st)).error(
                            "This is a patient account. Create a Doctor Sign Up account to access the doctor inbox."
                        )
                    elif user.license_number and user.license_number != doctor_license_number.strip():
                        _ = cast(_StreamlitErrorAPI, cast(object, st)).error(
                            "The medical license number does not match the doctor account."
                        )
                    else:
                        session_state["logged_in_user"] = user.username
                        session_state["user_role"] = "doctor"
                        session_state["doctor_certificate_name"] = user.certificate_name or str(getattr(doctor_certificate, "name", "certificate"))
                        session_state["doctor_license_number"] = user.license_number or doctor_license_number.strip()
                        DoctorAvailabilityStore().set_online(
                            user.username,
                            user.license_number or doctor_license_number.strip(),
                            user.certificate_name or str(getattr(doctor_certificate, "name", "certificate")),
                        )
                        _ = cast(_StreamlitSuccessAPI, cast(object, st)).success(
                            "Doctor credentials supplied. Opening the complaint inbox."
                        )
                        cast(Callable[[], object], getattr(cast(object, st), "rerun"))()

 
    # --- Patient Sign Up --------------------------------------------------
    with tabs[2]:
        with cast(_StreamlitFormFactoryAPI, cast(object, st)).form("patient_sign_up_form"):
            sign_up_username = cast(Callable[..., str], getattr(cast(object, st), "text_input"))("Choose a username", key="sign_up_username")
            sign_up_password = cast(Callable[..., str], getattr(cast(object, st), "text_input"))("Choose a password", type="password", key="sign_up_password")
            sign_up_confirm = cast(Callable[..., str], getattr(cast(object, st), "text_input"))("Confirm password", type="password", key="sign_up_confirm")
            sign_up_submitted = cast(Callable[..., bool], getattr(cast(object, st), "form_submit_button"))("Create Patient Account")
        if sign_up_submitted:
            try:
                user = user_store.register(sign_up_username, sign_up_password, sign_up_confirm)
            except (InvalidCredentialsError, UsernameTakenError) as exc:
                _ = cast(_StreamlitErrorAPI, cast(object, st)).error(str(exc))
            else:
                session_state["logged_in_user"] = user.username
                session_state["user_role"] = "user"
                session_state["doctor_certificate_name"] = ""
                session_state["doctor_license_number"] = ""
                _ = cast(_StreamlitSuccessAPI, cast(object, st)).success(f"Patient account created — welcome, {user.username}!")
                cast(Callable[[], object], getattr(cast(object, st), "rerun"))()

    # --- Doctor Sign Up --------------------------------------------------
    with tabs[3]:
        cast(Callable[[str], object], getattr(cast(object, st), "info"))(
            "Create a doctor account with your professional details. Certificate upload is a credential-submission step; this app does not independently verify it with a licensing authority."
        )
        with cast(_StreamlitFormFactoryAPI, cast(object, st)).form("doctor_sign_up_form"):
            doctor_signup_name = cast(Callable[..., str], getattr(cast(object, st), "text_input"))("Full professional name", key="doctor_signup_name")
            doctor_signup_specialty = cast(Callable[..., str], getattr(cast(object, st), "text_input"))("Medical specialty", key="doctor_signup_specialty")
            doctor_signup_license = cast(Callable[..., str], getattr(cast(object, st), "text_input"))("Medical license / certificate number", key="doctor_signup_license")
            doctor_signup_certificate = cast(Callable[..., object], getattr(cast(object, st), "file_uploader"))("Upload professional medical certificate", type=["pdf", "png", "jpg", "jpeg"], key="doctor_signup_certificate")
            doctor_signup_username = cast(Callable[..., str], getattr(cast(object, st), "text_input"))("Choose a doctor username", key="doctor_signup_username")
            doctor_signup_password = cast(Callable[..., str], getattr(cast(object, st), "text_input"))("Choose a password", type="password", key="doctor_signup_password")
            doctor_signup_confirm = cast(Callable[..., str], getattr(cast(object, st), "text_input"))("Confirm password", type="password", key="doctor_signup_confirm")
            doctor_signup_submitted = cast(Callable[..., bool], getattr(cast(object, st), "form_submit_button"))("Create Doctor Account")
        if doctor_signup_submitted:
            if doctor_signup_certificate is None:
                _ = cast(_StreamlitErrorAPI, cast(object, st)).error("Upload your professional medical certificate before creating the doctor account.")
            else:
                try:
                    doctor = user_store.register_doctor(doctor_signup_username, doctor_signup_password, doctor_signup_confirm, doctor_signup_name, doctor_signup_specialty, doctor_signup_license, str(getattr(doctor_signup_certificate, "name", "certificate")))
                except (InvalidCredentialsError, UsernameTakenError) as exc:
                    _ = cast(_StreamlitErrorAPI, cast(object, st)).error(str(exc))
                else:
                    session_state["logged_in_user"] = doctor.username
                    session_state["user_role"] = "doctor"
                    session_state["doctor_certificate_name"] = doctor.certificate_name
                    session_state["doctor_license_number"] = doctor.license_number
                    DoctorAvailabilityStore().set_online(doctor.username, doctor.license_number, doctor.certificate_name)
                    _ = cast(_StreamlitSuccessAPI, cast(object, st)).success(f"Doctor account created — welcome, Dr. {doctor.full_name}!")
                    cast(Callable[[], object], getattr(cast(object, st), "rerun"))()


def render_search_tab(history: SearchHistory, session_state: dict[str, object], api_key: str) -> None: 
    """Render the drug-lookup search box, results, recall banner, and 
    AI-simplified label sections. This is the app's original main 
    feature, extracted into its own function so it can live inside the 
    "🔍 Search" tab alongside the new Chat and Profile tabs.""" 
    if "drug_name_input" not in session_state: 
        session_state["drug_name_input"] = "" 
    if "trigger_search" not in session_state: 
        session_state["trigger_search"] = False 
 
    drug_name_input = cast( 
        Callable[..., str], getattr(cast(object, st), "text_input") 
    )( 
        "Enter a medication name", 
        placeholder="e.g. ibuprofen", 
        key="drug_name_input", 
    ) 
 
    # Live "did you mean" suggestions as the user types, powered by 
    # FDAClient.suggest_names(). Runs on every rerun (i.e. every 
    # keystroke that Streamlit picks up), but is cheap thanks to 
    # get_cached_suggestions() and openFDA's lightweight count-query mode. 
    render_search_suggestions(session_state, drug_name_input) 
 
    search_clicked = cast( 
        Callable[..., bool], getattr(cast(object, st), "button") 
    )("Search", type="primary") 
 
    if not (search_clicked or session_state.get("trigger_search")): 
        return 
 
    # A suggestion click already consumed itself by setting this flag — 
    # reset it so a plain rerun later (e.g. clearing history) doesn't 
    # re-trigger a search on its own. 
    session_state["trigger_search"] = False 
 
    # 1. Validate input 
    try: 
        drug_name = Medication.validate_name(drug_name_input) 
    except InvalidMedicationNameError as exc: 
        _ = cast(_StreamlitErrorAPI, cast(object, st)).error(str(exc)) 
        return 
 
    fda_client = FDAClient() 
    medication: Medication | None = None 
    recalls: list[dict[str, str]] = [] 
 
    # 2. Fetch label info 
    with cast(_StreamlitSpinnerFactoryAPI, cast(object, st)).spinner( 
        f"Looking up '{drug_name}'..." 
    ): 
        try: 
            medication = fda_client.fetch_label(drug_name) 
        except MedicationNotFoundError as exc: 
            _ = cast(_StreamlitWarningAPI, cast(object, st)).warning(str(exc)) 
            return 
        except FDANetworkError as exc: 
            _ = cast(_StreamlitErrorAPI, cast(object, st)).error( 
                f"Network problem while fetching drug info: {exc}" 
            ) 
            return 
 
        missing = medication.has_missing_fields() 
        if missing: 
            cast(Callable[[str], object], getattr(cast(object, st), "info"))( 
                "Note: the FDA label was missing data for: " + ", ".join(missing) 
            ) 
 
        # 3. Fetch recalls (non-fatal if it fails) 
        try: 
            recalls = fda_client.fetch_recalls(medication.generic_name or drug_name) 
        except FDANetworkError as exc: 
            _ = cast(_StreamlitWarningAPI, cast(object, st)).warning( 
                f"Could not check recalls right now: {exc}" 
            ) 
            recalls = [] 
 
    cast(Callable[[str], object], getattr(cast(object, st), "subheader"))( 
        f"{medication.generic_name.title() or drug_name.title()}" 
    ) 
    if medication.brand_names: 
        cast(Callable[[str], object], getattr(cast(object, st), "caption"))( 
            "Brand names: " + ", ".join(medication.brand_names) 
        ) 
 
    # 2b. Fetch a reference image (non-fatal if it fails) 
    image_client = ImageClient() 
    try: 
        image_url = image_client.fetch_thumbnail_url(medication, drug_name) 
        cast(Callable[..., object], getattr(cast(object, st), "image"))( 
            image_url, 
            caption=f"Image of {medication.generic_name or drug_name}", 
            width=300, 
        ) 
    except ImageNotFoundError: 
        cast(Callable[[str], object], getattr(cast(object, st), "caption"))( 
            "No reference image available for this medication." 
        ) 
 
    render_recall_banner(recalls) 
 
    keywords = medication.extract_warning_keywords() 
    if keywords: 
        cast(Callable[[str], object], getattr(cast(object, st), "markdown"))( 
            "**Key warning phrases (raw extract):**" 
        ) 
        _ = cast(_StreamlitWriteAPI, cast(object, st)).write( 
            " • " + "\n • ".join(keywords) 
        ) 
 
    # 4. Simplify with Gemini 
    simplified: dict[str, str] = {} 
    sections = { 
        "Usage": medication.usage, 
        "Dosage & Administration": medication.dosage, 
        "Warnings": medication.warnings, 
        "Side Effects": medication.side_effects, 
    } 
 
    markdown_fn = cast(Callable[[str], object], getattr(cast(object, st), "markdown")) 
    write_fn = cast(_StreamlitWriteAPI, cast(object, st)) 
 
    if not api_key: 
        _ = cast(_StreamlitWarningAPI, cast(object, st)).warning( 
            "No Gemini API key provided — showing raw FDA text instead of " 
            + "the simplified version." 
        ) 
        for label, text in sections.items(): 
            markdown_fn(f"### {label}") 
            write_fn.write(text or "_No data available._") 
    else: 
        translator = AITranslator(api_key=api_key) 
        for label, text in sections.items(): 
            markdown_fn(f"### {label}") 
            try: 
                with cast(_StreamlitSpinnerFactoryAPI, cast(object, st)).spinner( 
                    f"Simplifying '{label}'..." 
                ): 
                    simple_text = translator.simplify(text, label) 
                write_fn.write(simple_text) 
                simplified[label] = simple_text 
            except AITranslationError as exc: 
                _ = cast(_StreamlitErrorAPI, cast(object, st)).error( 
                    f"Could not simplify this section: {exc}" 
                ) 
                write_fn.write(text or "_No data available._") 
            time.sleep(0.2)  # gentle pacing between API calls 
 
    # 5. Save to history 
    entry = build_entry(drug_name, medication, recalls, simplified) 
    history.add(entry) 
    cast(Callable[[str], object], getattr(cast(object, st), "toast"))( 
        "Search saved to history." 
    ) 
 
 
def render_chat_tab(chat_history: ChatHistory, api_key: str) -> None: 
    """Render a free-form AI chat box (st.chat_message / st.chat_input) 
    backed by AITranslator.chat(), so users can ask general questions 
    beyond a single medication's label — with the conversation persisted 
    per-user via ChatHistory.""" 
    cast(Callable[[str], object], getattr(cast(object, st), "caption"))( 
        "Ask about medications, general health topics, or how to use this " 
        + "app. This chat is not a substitute for professional medical advice." 
    ) 
 
    chat_message_fn = cast( 
        Callable[[str], _StreamlitExpanderAPI], getattr(cast(object, st), "chat_message") 
    ) 
 
    messages = chat_history.get_all() 
    for msg in messages: 
        display_role = "assistant" if msg["role"] == "model" else "user" 
        with chat_message_fn(display_role): 
            cast(Callable[[str], object], getattr(cast(object, st), "markdown"))( 
                msg["content"] 
            ) 
 
    if messages and cast(Callable[..., bool], getattr(cast(object, st), "button"))( 
        "Clear chat", key="clear_chat_button" 
    ): 
        chat_history.clear() 
        cast(Callable[[], object], getattr(cast(object, st), "rerun"))() 
 
    chat_input_fn = cast(Callable[[str], str | None], getattr(cast(object, st), "chat_input")) 
    user_message = chat_input_fn("Ask something about your medications or health...") 
 
    if not user_message: 
        return 
 
    if not api_key: 
        _ = cast(_StreamlitErrorAPI, cast(object, st)).error( 
            "Add a Gemini API key in the sidebar to use the chat." 
        ) 
        return 
 
    prior_turns = chat_history.get_all()  # history BEFORE this new message 
    chat_history.add("user", user_message) 
 
    with chat_message_fn("user"): 
        cast(Callable[[str], object], getattr(cast(object, st), "markdown"))(user_message) 
 
    translator = AITranslator(api_key=api_key) 
    with chat_message_fn("assistant"): 
        try: 
            with cast(_StreamlitSpinnerFactoryAPI, cast(object, st)).spinner("Thinking..."): 
                reply = translator.chat(user_message, prior_turns) 
        except AITranslationError as exc: 
            reply = f"Sorry, I couldn't get a response right now: {exc}" 
        cast(Callable[[str], object], getattr(cast(object, st), "markdown"))(reply) 
 
    chat_history.add("model", reply) 
 
 
def render_patient_complaint_tab(
    api_key: str,
    session_state: dict[str, object],
    complaints: ComplaintStore,
    doctors: DoctorAvailabilityStore,
) -> None:
    """Let a patient submit a symptom complaint and route it to a doctor.

    Gemini produces a cautious, non-diagnostic triage summary. The original
    patient complaint is preserved for the doctor.
    """
    cast(Callable[[str], object], getattr(cast(object, st), "subheader"))(
        "📝 Patient Complaint"
    )
    cast(Callable[[str], object], getattr(cast(object, st), "caption"))(
        "Describe your symptoms or health concern. Your complaint can be routed "
        "to an available doctor. Do not include passwords, payment details, or "
        "other unnecessary personal information."
    )

    complaint_text = cast(
        Callable[..., str], getattr(cast(object, st), "text_area")
    )(
        "What are you experiencing?",
        placeholder=(
            "Example: I have had a fever and sore throat since yesterday. "
            "The fever is getting worse."
        ),
        height=180,
        key="patient_complaint_text",
    )

    emergency_note = cast(
        Callable[[str], object], getattr(cast(object, st), "warning")
    )
    emergency_note(
        "If you have severe trouble breathing, severe chest pain, loss of "
        "consciousness, uncontrolled bleeding, or another life-threatening "
        "emergency, seek emergency medical care immediately instead of waiting "
        "for a reply in this app."
    )

    submit = cast(
        Callable[..., bool], getattr(cast(object, st), "button")
    )("Send Complaint to an Available Doctor", type="primary")

    if submit:
        cleaned = complaint_text.strip()
        if len(cleaned) < 10:
            _ = cast(_StreamlitErrorAPI, cast(object, st)).error(
                "Please describe the complaint in a little more detail."
            )
            return
        if len(cleaned) > 5000:
            _ = cast(_StreamlitErrorAPI, cast(object, st)).error(
                "Please keep the complaint below 5,000 characters."
            )
            return
        if not api_key:
            _ = cast(_StreamlitErrorAPI, cast(object, st)).error(
                "Add a Gemini API key in the sidebar to use the cloud-AI complaint assistant."
            )
            return

        triage_prompt = (
            "You are a cautious medical information triage assistant. "
            "Review the patient's symptom complaint and create a short handoff "
            "summary for a licensed doctor. Do NOT diagnose, prescribe, or state "
            "that a disease is confirmed. Identify reported symptoms, duration "
            "or severity if supplied, important missing information the doctor "
            "may ask about, and whether the wording contains an obvious emergency "
            "warning sign. If an emergency may be present, say that urgent medical "
            "assessment is appropriate. Never invent facts.\n\n"
            f"Patient complaint:\n{cleaned}"
        )

        translator = AITranslator(api_key=api_key)
        try:
            with cast(_StreamlitSpinnerFactoryAPI, cast(object, st)).spinner(
                "Preparing your complaint for the doctor..."
            ):
                triage = translator.chat(triage_prompt, [])
        except AITranslationError as exc:
            _ = cast(_StreamlitErrorAPI, cast(object, st)).error(
                f"Cloud AI could not prepare the complaint: {exc}"
            )
            return

        patient_username = str(session_state.get("logged_in_user", ""))
        record = complaints.create(patient_username, cleaned, triage)

        # Automatically route to one currently available doctor.
        assigned = doctors.assign_available_doctor(str(record["id"]), complaints)
        if assigned:
            _ = cast(_StreamlitSuccessAPI, cast(object, st)).success(
                f"Complaint sent to available doctor: {assigned}."
            )
        else:
            _ = cast(_StreamlitSuccessAPI, cast(object, st)).success(
                "Complaint received. No doctor is currently marked available, "
                "so it has been placed in the shared doctor queue."
            )

    cast(Callable[[str], object], getattr(cast(object, st), "divider"))()
    cast(Callable[[str], object], getattr(cast(object, st), "subheader"))(
        "My complaints"
    )

    patient_username = str(session_state.get("logged_in_user", ""))
    mine = complaints.get_for_patient(patient_username)
    if not mine:
        cast(_StreamlitWriteAPI, cast(object, st)).write(
            "You have not submitted a complaint yet."
        )
        return

    for item in mine[:20]:
        status = str(item.get("status", "waiting")).replace("_", " ").title()
        created = str(item.get("created_at", ""))
        with cast(_StreamlitExpanderFactoryAPI, cast(object, st)).expander(
            f"{status} — {created}"
        ):
            cast(_StreamlitWriteAPI, cast(object, st)).write(
                f"**Your complaint:** {item.get('complaint', '')}"
            )
            assigned = str(item.get("assigned_doctor", ""))
            cast(_StreamlitWriteAPI, cast(object, st)).write(
                f"**Doctor:** {assigned or 'Waiting for an available doctor'}"
            )
            if item.get("doctor_response"):
                cast(_StreamlitWriteAPI, cast(object, st)).write(
                    f"**Doctor response:** {item['doctor_response']}"
                )
            cast(_StreamlitWriteAPI, cast(object, st)).write(
                "**AI handoff summary:**"
            )
            cast(Callable[[str], object], getattr(cast(object, st), "markdown"))(
                str(item.get("triage", "No AI summary available."))
            )


def render_doctor_inbox_tab(
    session_state: dict[str, object],
    complaints: ComplaintStore,
    doctors: DoctorAvailabilityStore,
) -> None:
    """Show doctors the shared patient complaint queue.

    This replaces the former doctor patient-care workspace.
    """
    doctor_username = str(session_state.get("logged_in_user", ""))

    cast(Callable[[str], object], getattr(cast(object, st), "subheader"))(
        "📥 Doctor Complaint Inbox"
    )
    cast(Callable[[str], object], getattr(cast(object, st), "caption"))(
        "Patient complaints are routed here. Review the patient's original "
        "description, the AI-generated handoff summary, and respond using your "
        "professional judgment."
    )

    available = doctors.available_doctors()
    _ = cast(_StreamlitSuccessAPI, cast(object, st)).success(
        f"{len(available)} doctor(s) currently marked available."
    )

    waiting = complaints.get_waiting()
    assigned_to_me = complaints.get_for_doctor(doctor_username)

    cast(Callable[[str], object], getattr(cast(object, st), "markdown"))(
        f"**Waiting in queue:** {len(waiting)}  \n"
        f"**Assigned to you:** {len(assigned_to_me)}"
    )

    if waiting:
        cast(Callable[[str], object], getattr(cast(object, st), "divider"))()
        cast(Callable[[str], object], getattr(cast(object, st), "subheader"))(
            "Waiting complaints"
        )
        for item in waiting[:20]:
            complaint_id = str(item.get("id", ""))
            with cast(_StreamlitExpanderFactoryAPI, cast(object, st)).expander(
                f"Complaint from {item.get('patient_username', 'patient')} — "
                f"{item.get('created_at', '')}"
            ):
                cast(_StreamlitWriteAPI, cast(object, st)).write(
                    f"**Patient complaint:** {item.get('complaint', '')}"
                )
                cast(_StreamlitWriteAPI, cast(object, st)).write(
                    "**Cloud-AI handoff summary:**"
                )
                cast(Callable[[str], object], getattr(cast(object, st), "markdown"))(
                    str(item.get("triage", "No AI summary available."))
                )
                if cast(
                    Callable[..., bool], getattr(cast(object, st), "button")
                )("Claim complaint", key=f"claim_{complaint_id}"):
                    if complaints.claim(complaint_id, doctor_username):
                        _ = cast(_StreamlitSuccessAPI, cast(object, st)).success(
                            "Complaint claimed. Refreshing your inbox."
                        )
                        cast(Callable[[], object], getattr(cast(object, st), "rerun"))()
                    else:
                        _ = cast(_StreamlitErrorAPI, cast(object, st)).error(
                            "This complaint was already claimed by another doctor."
                        )

    if assigned_to_me:
        cast(Callable[[str], object], getattr(cast(object, st), "divider"))()
        cast(Callable[[str], object], getattr(cast(object, st), "subheader"))(
            "My assigned complaints"
        )
        for item in assigned_to_me[:20]:
            complaint_id = str(item.get("id", ""))
            with cast(_StreamlitExpanderFactoryAPI, cast(object, st)).expander(
                f"{item.get('status', 'claimed').title()} — "
                f"{item.get('created_at', '')}"
            ):
                cast(_StreamlitWriteAPI, cast(object, st)).write(
                    f"**Patient complaint:** {item.get('complaint', '')}"
                )
                cast(_StreamlitWriteAPI, cast(object, st)).write(
                    "**Cloud-AI handoff summary:**"
                )
                cast(Callable[[str], object], getattr(cast(object, st), "markdown"))(
                    str(item.get("triage", "No AI summary available."))
                )
                response = cast(
                    Callable[..., str], getattr(cast(object, st), "text_area")
                )(
                    "Response to patient",
                    value=str(item.get("doctor_response", "")),
                    key=f"response_{complaint_id}",
                    height=140,
                    placeholder=(
                        "Provide appropriate medical guidance, explain when the "
                        "patient should seek urgent care, and avoid claiming a "
                        "diagnosis unless clinically established outside this app."
                    ),
                )
                if cast(
                    Callable[..., bool], getattr(cast(object, st), "button")
                )("Send response to patient", key=f"send_response_{complaint_id}"):
                    if complaints.respond(complaint_id, doctor_username, response):
                        _ = cast(_StreamlitSuccessAPI, cast(object, st)).success(
                            "Response sent to the patient."
                        )
                        cast(Callable[[], object], getattr(cast(object, st), "rerun"))()
                    else:
                        _ = cast(_StreamlitErrorAPI, cast(object, st)).error(
                            "Could not send the response. Make sure you entered a response."
                        )

    _ = cast(_StreamlitWarningAPI, cast(object, st)).warning(
        "The AI summary is not a diagnosis and may be incomplete. Doctors remain "
        "responsible for clinical assessment, decisions, and emergency escalation."
    )



def render_profile_tab( 
    user: User, 
    search_entries: list[dict[str, object]], 
    user_store: UserStore, 
) -> None: 
    """Render account info, usage stats, and a change-password form.""" 
    cast(Callable[[str], object], getattr(cast(object, st), "subheader"))( 
        f"👤 {user.username}" 
    ) 
    cast(_StreamlitWriteAPI, cast(object, st)).write( 
        f"**Member since:** {user.created_at or 'Unknown'}" 
    ) 
 
    stats = compute_profile_stats(search_entries) 
    columns_fn = cast(Callable[[int], list[object]], getattr(cast(object, st), "columns")) 
    metric_fn = cast(Callable[..., object], getattr(cast(object, st), "metric")) 
    stat_columns = columns_fn(2) 
    with cast(_StreamlitExpanderAPI, stat_columns[0]): 
        metric_fn("Total searches", stats["total_searches"]) 
    with cast(_StreamlitExpanderAPI, stat_columns[1]): 
        metric_fn("Recalls flagged", stats["recalls_flagged"]) 
 
    cast(Callable[[], object], getattr(cast(object, st), "divider"))() 
    cast(Callable[[str], object], getattr(cast(object, st), "subheader"))( 
        "Change password" 
    ) 
 
    with cast(_StreamlitFormFactoryAPI, cast(object, st)).form("change_password_form"): 
        current_password = cast( 
            Callable[..., str], getattr(cast(object, st), "text_input") 
        )("Current password", type="password", key="profile_current_password") 
        new_password = cast( 
            Callable[..., str], getattr(cast(object, st), "text_input") 
        )("New password", type="password", key="profile_new_password") 
        confirm_new_password = cast( 
            Callable[..., str], getattr(cast(object, st), "text_input") 
        )("Confirm new password", type="password", key="profile_confirm_new_password") 
        password_submitted = cast( 
            Callable[..., bool], getattr(cast(object, st), "form_submit_button") 
        )("Update password") 
 
    if password_submitted: 
        try: 
            user_store.change_password( 
                user.username, current_password, new_password, confirm_new_password 
            ) 
        except InvalidCredentialsError as exc: 
            _ = cast(_StreamlitErrorAPI, cast(object, st)).error(str(exc)) 
        else: 
            _ = cast(_StreamlitSuccessAPI, cast(object, st)).success( 
                "Password updated." 
            ) 
 
 
def main() -> None: 
    _ = cast(_StreamlitPageConfigAPI, cast(object, st)).set_page_config( 
        page_title="MedSimplify", page_icon="💊", layout="centered" 
    ) 
    _ = cast(Callable[[str], object], getattr(cast(object, st), "title"))( 
        "💊 MedSimplify" 
    ) 
    _ = cast(Callable[[str], object], getattr(cast(object, st), "caption"))( 
        "Look up medication information, check active recalls, or use the " 
        + "patient complaint service, or the doctor complaint inbox when signed in as a doctor." 
    ) 
 
    # --- Authentication gate --------------------------------------------- 
    # Everything below requires a signed-in user. session_state persists 
    # across Streamlit reruns (every click reruns the whole script), 
    # which is what lets a user stay "logged in" between actions. 
    session_state = cast(dict[str, object], getattr(cast(object, st), "session_state")) 
    if "logged_in_user" not in session_state: 
        session_state["logged_in_user"] = None 
    if "user_role" not in session_state: 
        session_state["user_role"] = "user" 
    if "doctor_certificate_name" not in session_state: 
        session_state["doctor_certificate_name"] = "" 
    if "doctor_license_number" not in session_state: 
        session_state["doctor_license_number"] = "" 
 
    user_store = UserStore() 
 
    if not session_state.get("logged_in_user"): 
        render_auth_ui(user_store, session_state) 
        return 
 
    logged_in_user = cast(str, session_state["logged_in_user"]) 
 
    # --- Sidebar: account + API key + history ----------------------------- 
    sidebar = cast( 
        _StreamlitSidebarAPI, 
        getattr(cast(object, st), "sidebar"), 
    ) 
    # Gemini API key: use Streamlit Secrets or the GEMINI_API_KEY environment variable.
    # Do not hard-code a key in source code; keys exposed in source/chat may be revoked.
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
        if isinstance(secret_key, str) and secret_key.strip():
            api_key = secret_key.strip()
    except Exception:
        pass
    # Each user's search/chat history lives in its own file, keyed by a 
    # lowercased/sanitized version of their username, so accounts don't 
    # share or overwrite each other's saved data. 
    safe_username = re.sub(r"[^a-z0-9_-]", "_", logged_in_user.lower()) 
    history = SearchHistory(filepath=f"search_history_{safe_username}.json") 
    chat_history = ChatHistory(filepath=f"chat_history_{safe_username}.json") 
    with sidebar: 
        _ = cast(Callable[[str], object], getattr(cast(object, st), "write"))( 
            f"👤 Signed in as **{logged_in_user}**" 
        ) 
        if session_state.get("user_role") == "doctor": 
            _ = cast(Callable[[str], object], getattr(cast(object, st), "success"))( 
                "🩺 Doctor access" 
            ) 
        if cast(Callable[[str], bool], getattr(cast(object, st), "button"))( 
            "Log out" 
        ): 
            if session_state.get("user_role") == "doctor":
                DoctorAvailabilityStore().set_offline(logged_in_user)
            session_state["logged_in_user"] = None 
            session_state["user_role"] = "user" 
            session_state["doctor_certificate_name"] = "" 
            session_state["doctor_license_number"] = "" 
            cast(Callable[[], object], getattr(cast(object, st), "rerun"))() 
 
        _ = cast(Callable[[], object], getattr(cast(object, st), "divider"))() 
        _ = cast(Callable[[str,], object], getattr(cast(object, st), "header"))( 
            "Settings" 
        ) 
        _ = cast(_StreamlitWriteAPI, cast(object, st)).write(
            "🤖 Cloud AI is enabled."
        )
 
        _ = cast(Callable[[], object], getattr(cast(object, st), "divider"))() 
        _ = cast(Callable[[str], object], getattr(cast(object, st), "header"))( 
            "Search History" 
        ) 
        entries = history.get_all() 
        if entries: 
            for e in entries[:15]: 
                flag = "⚠️" if e["recall_found"] else "" 
                _ = cast(_StreamlitWriteAPI, cast(object, st)).write( 
                    f"{flag} **{e['query']}** — {e['timestamp']}" 
                ) 
            if cast(Callable[[str], bool], getattr(cast(object, st), "button"))( 
                "Clear history" 
            ): 
                history.clear() 
                _ = cast(Callable[[], object], getattr(cast(object, st), "rerun"))() 
        else: 
            _ = cast(_StreamlitWriteAPI, cast(object, st)).write( 
                "No searches yet." 
            ) 
 
    # --- Main content: patient complaint / doctor inbox / profile ----------
    if session_state.get("user_role") == "doctor":
        main_tabs = cast(_StreamlitTabsFactoryAPI, cast(object, st)).tabs(
            ["🔍 Medication Search", "📥 Doctor Inbox", "👤 Profile"]
        )

        with main_tabs[0]:
            render_search_tab(history, session_state, api_key)

        with main_tabs[1]:
            render_doctor_inbox_tab(
                session_state,
                ComplaintStore(),
                DoctorAvailabilityStore(),
            )

        profile_tab = main_tabs[2]
    else:
        main_tabs = cast(_StreamlitTabsFactoryAPI, cast(object, st)).tabs(
            ["🔍 Medication Search", "📝 Complaint", "👤 Profile"]
        )

        with main_tabs[0]:
            render_search_tab(history, session_state, api_key)

        with main_tabs[1]:
            render_patient_complaint_tab(
                api_key,
                session_state,
                ComplaintStore(),
                DoctorAvailabilityStore(),
            )

        profile_tab = main_tabs[2]

    with profile_tab: 
        current_user = user_store.get_user(logged_in_user) 
        if current_user is None: 
            # Account was deleted from users.json out from under an 
            # active session — log the user out rather than crash. 
            session_state["logged_in_user"] = None 
            _ = cast(_StreamlitErrorAPI, cast(object, st)).error( 
                "Your account could not be found. Please sign in again." 
            ) 
            cast(Callable[[], object], getattr(cast(object, st), "rerun"))() 
        else: 
            render_profile_tab(current_user, history.get_all(), user_store) 
 
 
if __name__ == "__main__":
    main()
