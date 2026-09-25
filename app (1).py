from **future** import annotations

import hashlib
import importlib
import json
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

# ---------------------------------------------------------------------------

# Types / protocols

# ---------------------------------------------------------------------------

class _HTTPResponse(Protocol):
status_code: int
ok: bool
text: str

```
def json(self) -> object:
    ...
```

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
def **enter**(self) -> "_StreamlitExpanderAPI":
...

```
def __exit__(
    self,
    exc_type: object,
    exc_value: object,
    traceback: object,
) -> bool:
    ...
```

class _StreamlitExpanderFactoryAPI(Protocol):
def expander(self, label: str) -> _StreamlitExpanderAPI:
...

class _StreamlitSpinnerAPI(Protocol):
def **enter**(self) -> "_StreamlitSpinnerAPI":
...

```
def __exit__(
    self,
    exc_type: object,
    exc_value: object,
    traceback: object,
) -> bool:
    ...
```

class _StreamlitSpinnerFactoryAPI(Protocol):
def spinner(self, text: str) -> _StreamlitSpinnerAPI:
...

class _StreamlitPageConfigAPI(Protocol):
def set_page_config(
self,
*,
page_title: str,
page_icon: str,
layout: str,
) -> object:
...

class _StreamlitSidebarAPI(Protocol):
def **enter**(self) -> "_StreamlitSidebarAPI":
...

```
def __exit__(
    self,
    exc_type: object,
    exc_value: object,
    traceback: object,
) -> bool:
    ...

def header(self, body: str) -> object:
    ...

def text_input(
    self,
    label: str,
    *,
    value: str,
    type: str,
    help: str,
) -> str:
    ...

def divider(self) -> object:
    ...

def write(self, body: str) -> object:
    ...

def button(self, label: str) -> bool:
    ...
```

class _StreamlitFormAPI(Protocol):
def **enter**(self) -> "_StreamlitFormAPI":
...

```
def __exit__(
    self,
    exc_type: object,
    exc_value: object,
    traceback: object,
) -> bool:
    ...
```

class _StreamlitFormFactoryAPI(Protocol):
def form(
self,
key: str,
*,
clear_on_submit: bool = ...,
) -> _StreamlitFormAPI:
...

class _StreamlitTabsFactoryAPI(Protocol):
def tabs(self, labels: list[str]) -> list[object]:
...

requests: ModuleType = importlib.import_module("requests")
RequestException = cast(
_RequestsModule,
cast(object, requests),
).exceptions.RequestException

try:
st: ModuleType = importlib.import_module("streamlit")
except ModuleNotFoundError:

```
class _StreamlitFallback:
    def __call__(self, *args: object, **kwargs: object) -> None:
        return None

    def __getattr__(self, name: str) -> "_StreamlitFallback":
        return self

    def __enter__(self) -> "_StreamlitFallback":
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> bool:
        return False

st = cast(ModuleType, cast(object, _StreamlitFallback()))
```

# ---------------------------------------------------------------------------

# Custom exceptions

# ---------------------------------------------------------------------------

class MedSimplifyError(Exception):
"""Base class for all app-specific errors."""

class InvalidMedicationNameError(MedSimplifyError):
"""Raised when the medication name is invalid."""

class MedicationNotFoundError(MedSimplifyError):
"""Raised when no FDA label is found."""

class FDANetworkError(MedSimplifyError):
"""Raised when an openFDA request fails."""

class AITranslationError(MedSimplifyError):
"""Raised when Gemini fails."""

class ImageNotFoundError(MedSimplifyError):
"""Raised when no medication image is available."""

class AuthenticationError(MedSimplifyError):
"""Base authentication error."""

class UsernameTakenError(AuthenticationError):
"""Raised when a username already exists."""

class InvalidCredentialsError(AuthenticationError):
"""Raised when authentication fails."""

# ---------------------------------------------------------------------------

# Medication

# ---------------------------------------------------------------------------

@dataclass
class Medication:
name: str
generic_name: str = ""
brand_names: list[str] = field(default_factory=list)
usage: str = ""
warnings: str = ""
side_effects: str = ""
dosage: str = ""
raw: dict[str, object] = field(default_factory=dict)

```
_CITATION_RE: ClassVar[re.Pattern[str]] = re.compile(
    r"\(\d+(?:\.\d+)*\)"
)

_WHITESPACE_RE: ClassVar[re.Pattern[str]] = re.compile(
    r"\s+"
)

_BULLET_RE: ClassVar[re.Pattern[str]] = re.compile(
    r"^\s*[\u2022\-\*]\s*",
    re.MULTILINE,
)

_NAME_VALIDATION_RE: ClassVar[re.Pattern[str]] = re.compile(
    r"^[A-Za-z0-9\s\-/]+$"
)

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
    cleaned = name.strip()

    if not cleaned:
        raise InvalidMedicationNameError(
            "Medication name cannot be empty."
        )

    if len(cleaned) > 100:
        raise InvalidMedicationNameError(
            "Medication name is too long."
        )

    if not cls._NAME_VALIDATION_RE.match(cleaned):
        raise InvalidMedicationNameError(
            "Medication name may only contain letters, numbers, "
            "spaces, hyphens, and slashes."
        )

    return cleaned

@classmethod
def clean_text(
    cls,
    text: str | list[str] | None,
) -> str:
    if not text:
        return ""

    if isinstance(text, list):
        text = " ".join(text)

    text = cls._CITATION_RE.sub("", text)
    text = cls._BULLET_RE.sub("", text)
    text = cls._WHITESPACE_RE.sub(" ", text)

    return text.strip()

@classmethod
def from_openfda_result(
    cls,
    name: str,
    result: dict[str, object],
) -> "Medication":

    openfda_value = result.get("openfda")

    openfda: dict[str, object] = {}

    if isinstance(openfda_value, dict):
        for key, value in cast(
            dict[object, object],
            openfda_value,
        ).items():
            if isinstance(key, str):
                openfda[key] = value

    def text_field(
        key: str,
    ) -> str | list[str] | None:

        value = result.get(key)

        if isinstance(value, str):
            return value

        if isinstance(value, list):
            return [
                item
                for item in cast(list[object], value)
                if isinstance(item, str)
            ]

        return None

    def string_list_field(key: str) -> list[str]:
        value = openfda.get(key)

        if isinstance(value, list):
            return [
                item
                for item in cast(list[object], value)
                if isinstance(item, str)
            ]

        return []

    generic_names = string_list_field("generic_name")
    brand_names = string_list_field("brand_name")

    warnings = (
        text_field("warnings")
        or text_field("warnings_and_cautions")
    )

    return cls(
        name=name,
        generic_name=", ".join(generic_names) or name,
        brand_names=brand_names,
        usage=cls.clean_text(
            text_field("indications_and_usage")
        ),
        warnings=cls.clean_text(warnings),
        side_effects=cls.clean_text(
            text_field("adverse_reactions")
        ),
        dosage=cls.clean_text(
            text_field("dosage_and_administration")
        ),
        raw=result,
    )

def extract_warning_keywords(self) -> list[str]:
    if not self.warnings:
        return []

    found: list[str] = []

    for pattern in self._WARNING_KEYWORD_PATTERNS:
        matches = re.findall(
            pattern,
            self.warnings,
            flags=re.IGNORECASE,
        )

        for match in matches:
            snippet = match.strip()

            if snippet and snippet not in found:
                found.append(snippet[:160])

    return found[:8]

def has_missing_fields(self) -> list[str]:
    missing: list[str] = []

    for attr_name in (
        "usage",
        "warnings",
        "side_effects",
        "dosage",
    ):
        if not getattr(self, attr_name):
            missing.append(attr_name)

    return missing
```

# ---------------------------------------------------------------------------

# FDA Client

# ---------------------------------------------------------------------------

class FDAClient:
LABEL_URL = "https://api.fda.gov/drug/label.json"
ENFORCEMENT_URL = (
"https://api.fda.gov/drug/enforcement.json"
)

```
def __init__(self, timeout: int = 10):
    self.timeout = timeout

def _get(
    self,
    url: str,
    params: dict[str, str],
) -> dict[str, object]:

    try:
        response = cast(
            Callable[..., _HTTPResponse],
            requests.get,
        )(
            url,
            params=params,
            timeout=self.timeout,
        )

    except RequestException as exc:
        raise FDANetworkError(
            f"Could not reach openFDA: {exc}"
        ) from exc

    if response.status_code == 404:
        return {"results": []}

    if not response.ok:
        raise FDANetworkError(
            f"openFDA returned an error "
            f"(status {response.status_code})."
        )

    try:
        data = response.json()

    except ValueError as exc:
        raise FDANetworkError(
            "openFDA returned an unreadable response."
        ) from exc

    if not isinstance(data, dict):
        raise FDANetworkError(
            "openFDA returned an unreadable response."
        )

    return cast(dict[str, object], data)

def fetch_label(
    self,
    drug_name: str,
) -> Medication:

    queries = [
        f'openfda.generic_name:"{drug_name}"',
        f'openfda.brand_name:"{drug_name}"',
        f'indications_and_usage:"{drug_name}"',
    ]

    for query in queries:

        data = self._get(
            self.LABEL_URL,
            {
                "search": query,
                "limit": "1",
            },
        )

        results_value = data.get("results")

        if not isinstance(results_value, list):
            continue

        for candidate in cast(
            list[object],
            results_value,
        ):
            if isinstance(candidate, dict):
                return Medication.from_openfda_result(
                    drug_name,
                    cast(dict[str, object], candidate),
                )

    raise MedicationNotFoundError(
        f'No FDA label information found for "{drug_name}". '
        "Check the spelling or try the generic name."
    )

def suggest_names(
    self,
    partial_name: str,
    limit: int = 8,
) -> list[str]:

    cleaned = partial_name.strip()

    if len(cleaned) < 2:
        return []

    suggestions: list[str] = []
    seen_lower: set[str] = set()

    for count_field in (
        "openfda.generic_name.exact",
        "openfda.brand_name.exact",
    ):

        query = f"{count_field}:{cleaned}*"

        try:
            data = self._get(
                self.LABEL_URL,
                {
                    "search": query,
                    "count": count_field,
                    "limit": str(limit),
                },
            )

        except FDANetworkError:
            continue

        results_value = data.get("results")

        if not isinstance(results_value, list):
            continue

        for item in cast(
            list[object],
            results_value,
        ):

            if not isinstance(item, dict):
                continue

            term = cast(
                dict[str, object],
                item,
            ).get("term")

            if not isinstance(term, str):
                continue

            normalized = term.strip().title()

            if (
                normalized
                and normalized.lower() not in seen_lower
            ):
                seen_lower.add(normalized.lower())
                suggestions.append(normalized)

    return suggestions[:limit]

def fetch_recalls(
    self,
    drug_name: str,
    limit: int = 5,
) -> list[dict[str, str]]:

    query = f'product_description:"{drug_name}"'

    data = self._get(
        self.ENFORCEMENT_URL,
        {
            "search": query,
            "limit": str(limit),
            "sort": "report_date:desc",
        },
    )

    results_value = data.get("results")

    if not isinstance(results_value, list):
        return []

    recalls: list[dict[str, str]] = []

    for item in cast(
        list[object],
        results_value,
    ):

        if not isinstance(item, dict):
            continue

        item_dict = cast(
            dict[str, object],
            item,
        )

        product_description = item_dict.get(
            "product_description"
        )

        reason_for_recall = item_dict.get(
            "reason_for_recall"
        )

        classification = item_dict.get(
            "classification",
            "Unknown",
        )

        status = item_dict.get(
            "status",
            "Unknown",
        )

        report_date = item_dict.get(
            "report_date",
            "",
        )

        recalling_firm = item_dict.get(
            "recalling_firm",
            "",
        )

        def convert_value(value: object) -> str:

            if isinstance(value, str):
                return value

            if isinstance(value, list):
                return " ".join(
                    str(part)
                    for part in cast(list[object], value)
                    if isinstance(part, str)
                )

            return ""

        recalls.append(
            {
                "product_description": Medication.clean_text(
                    convert_value(product_description)
                ),
                "reason_for_recall": Medication.clean_text(
                    convert_value(reason_for_recall)
                ),
                "classification": str(classification),
                "status": str(status),
                "report_date": str(report_date),
                "recalling_firm": str(recalling_firm),
            }
        )

    return recalls
```

# ---------------------------------------------------------------------------

# Gemini translator

# ---------------------------------------------------------------------------

class AITranslator:
API_URL_TEMPLATE = (
"https://generativelanguage.googleapis.com/v1beta/"
"models/{model}:generateContent"
)

```
def __init__(
    self,
    api_key: str,
    model: str = "gemini-2.0-flash",
):
    if not api_key:
        raise AITranslationError(
            "No Gemini API key was provided."
        )

    self.api_key = api_key
    self.model = model

def simplify(
    self,
    text: str,
    section_name: str,
) -> str:

    if not text:
        return (
            "No information was provided by the FDA "
            "for this section."
        )

    prompt = (
        "You are helping a patient with no medical background "
        "understand their medication. Rewrite the following "
        f"'{section_name}' section from an official FDA drug "
        "label in simple, clear, everyday language. "
        "Keep it accurate, use short sentences, and avoid "
        "jargon. Limit your answer to about 120 words.\n\n"
        f"Original text:\n{text}"
    )

    url = self.API_URL_TEMPLATE.format(
        model=self.model
    )

    request_payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }

    try:
        response = cast(
            Callable[..., _HTTPResponse],
            requests.post,
        )(
            url,
            headers={
                "Content-Type": "application/json"
            },
            params={
                "key": self.api_key
            },
            data=json.dumps(request_payload),
            timeout=30,
        )

    except RequestException as exc:
        raise AITranslationError(
            f"Could not reach Gemini API: {exc}"
        ) from exc

    if not response.ok:
        raise AITranslationError(
            f"Gemini API returned status "
            f"{response.status_code}: "
            f"{response.text[:200]}"
        )

    try:
        response_payload = response.json()

    except ValueError as exc:
        raise AITranslationError(
            "Gemini API returned an unexpected response."
        ) from exc

    return self._extract_text(response_payload)

@staticmethod
def _extract_text(
    response_payload: object,
) -> str:

    if not isinstance(response_payload, dict):
        raise AITranslationError(
            "Gemini API returned an unexpected response."
        )

    candidates = cast(
        dict[str, object],
        response_payload,
    ).get("candidates")

    if not isinstance(candidates, list) or not candidates:
        raise AITranslationError(
            "Gemini API returned no answer."
        )

    first_candidate = cast(
        list[object],
        candidates,
    )[0]

    if not isinstance(first_candidate, dict):
        raise AITranslationError(
            "Gemini API returned an unexpected response."
        )

    content = cast(
        dict[str, object],
        first_candidate,
    ).get("content")

    if not isinstance(content, dict):
        raise AITranslationError(
            "Gemini API returned an unexpected response."
        )

    parts = cast(
        dict[str, object],
        content,
    ).get("parts")

    if not isinstance(parts, list) or not parts:
        raise AITranslationError(
            "Gemini API returned an unexpected response."
        )

    first_part = cast(
        list[object],
        parts,
    )[0]

    if not isinstance(first_part, dict):
        raise AITranslationError(
            "Gemini API returned an unexpected response."
        )

    text_value = cast(
        dict[str, object],
        first_part,
    ).get("text")

    if not isinstance(text_value, str):
        raise AITranslationError(
            "Gemini API returned an unexpected response."
        )

    return text_value.strip()
```

# ---------------------------------------------------------------------------

# AI Chat

# ---------------------------------------------------------------------------

class AIChat:
"""Medication-specific conversational Gemini assistant."""

```
API_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/{model}:generateContent"
)

def __init__(
    self,
    api_key: str,
    model: str = "gemini-2.0-flash",
):
    if not api_key:
        raise AITranslationError(
            "No Gemini API key was provided."
        )

    self.api_key = api_key
    self.model = model

def ask(
    self,
    medication: Medication,
    question: str,
    conversation: list[dict[str, str]],
) -> str:

    if not question.strip():
        return "Please enter a question."

    context = f"""
```

MEDICATION:
{medication.generic_name or medication.name}

BRAND NAMES:
{", ".join(medication.brand_names) or "None listed"}

FDA USAGE:
{medication.usage or "Not provided"}

FDA DOSAGE:
{medication.dosage or "Not provided"}

FDA WARNINGS:
{medication.warnings or "Not provided"}

FDA SIDE EFFECTS:
{medication.side_effects or "Not provided"}
"""

```
    conversation_text = ""

    for message in conversation[-10:]:
        role = message.get("role", "")
        content = message.get("content", "")

        if role and content:
            conversation_text += (
                f"\n{role.upper()}: {content}\n"
            )

    system_instruction = """
```

You are MedSimplify AI, a medication information assistant.

Your job is to help users understand FDA medication label
information in plain, easy-to-understand language.

IMPORTANT RULES:

1. Use the supplied FDA label information as your primary source.
2. Do not invent facts that are not in the supplied information.
3. Explain medical terminology in everyday language.
4. Do not diagnose the user.
5. Do not prescribe medication.
6. Do not tell a user to start, stop, increase, or decrease a dose.
7. If the user asks for personalized medical advice, explain that
   a doctor or pharmacist should provide individualized advice.
8. If the user describes a possible medical emergency, encourage
   them to seek urgent medical attention.
9. If the answer is not contained in the supplied label context,
   say that the available label information does not answer it.
10. Keep answers concise and clear.
    """

    ```
    prompt = (
        f"{system_instruction}\n\n"
        f"CURRENT FDA MEDICATION CONTEXT:\n"
        f"{context}\n\n"
        f"RECENT CHAT:\n"
        f"{conversation_text}\n\n"
        f"NEW USER QUESTION:\n{question}\n\n"
        "Answer the user's question."
    )

    url = self.API_URL_TEMPLATE.format(
        model=self.model
    )

    request_payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }

    try:
        response = cast(
            Callable[..., _HTTPResponse],
            requests.post,
        )(
            url,
            headers={
                "Content-Type": "application/json"
            },
            params={
                "key": self.api_key
            },
            data=json.dumps(request_payload),
            timeout=30,
        )

    except RequestException as exc:
        raise AITranslationError(
            f"Could not reach Gemini API: {exc}"
        ) from exc

    if not response.ok:
        raise AITranslationError(
            f"Gemini API returned status "
            f"{response.status_code}: "
            f"{response.text[:200]}"
        )

    try:
        response_payload = response.json()

    except ValueError as exc:
        raise AITranslationError(
            "Gemini returned an unreadable response."
        ) from exc

    return AITranslator._extract_text(
        response_payload
    )
    ```

# ---------------------------------------------------------------------------

# Image client

# ---------------------------------------------------------------------------

class ImageClient:
SUMMARY_URL_TEMPLATE = (
"https://en.wikipedia.org/api/rest_v1/page/summary/"
"{title}"
)

```
USER_AGENT = (
    "MedSimplify/1.0 "
    "(educational project; contact: example@example.com)"
)

def __init__(self, timeout: int = 10):
    self.timeout = timeout

def _fetch_thumbnail_for_title(
    self,
    title: str,
) -> str | None:

    url = self.SUMMARY_URL_TEMPLATE.format(
        title=title.strip().replace(" ", "_")
    )

    try:
        response = cast(
            Callable[..., _HTTPResponse],
            requests.get,
        )(
            url,
            headers={
                "User-Agent": self.USER_AGENT
            },
            timeout=self.timeout,
        )

    except RequestException as exc:
        raise ImageNotFoundError(
            f"Could not reach Wikipedia: {exc}"
        ) from exc

    if response.status_code == 404:
        return None

    if not response.ok:
        raise ImageNotFoundError(
            f"Wikipedia returned an error "
            f"(status {response.status_code})."
        )

    try:
        data = response.json()

    except ValueError as exc:
        raise ImageNotFoundError(
            "Wikipedia returned an unreadable response."
        ) from exc

    if not isinstance(data, dict):
        return None

    thumbnail = cast(
        dict[str, object],
        data,
    ).get("thumbnail")

    if not isinstance(thumbnail, dict):
        return None

    source = cast(
        dict[str, object],
        thumbnail,
    ).get("source")

    if not isinstance(source, str):
        return None

    return source

def fetch_thumbnail_url(
    self,
    medication: Medication,
    fallback_name: str,
) -> str:

    candidates: list[str] = []

    if medication.generic_name:
        candidates.append(
            medication.generic_name
        )

    candidates.extend(
        medication.brand_names
    )

    if fallback_name not in candidates:
        candidates.append(fallback_name)

    for candidate in candidates:
        thumbnail = self._fetch_thumbnail_for_title(
            candidate
        )

        if thumbnail:
            return thumbnail

    raise ImageNotFoundError(
        f'No image available for "{fallback_name}".'
    )
```

# ---------------------------------------------------------------------------

# Users

# ---------------------------------------------------------------------------

@dataclass
class User:
username: str
password_hash: str
salt: str
created_at: str

class UserStore:
_HASH_ITERATIONS: ClassVar[int] = 200_000
_MIN_PASSWORD_LENGTH: ClassVar[int] = 8

```
def __init__(
    self,
    filepath: str = "users.json",
):
    self.filepath = Path(filepath)

    if not self.filepath.exists():
        self._write({})

def _read(
    self,
) -> dict[str, dict[str, object]]:

    try:
        with self.filepath.open(
            "r",
            encoding="utf-8",
        ) as f:
            loaded = json.load(f)

    except (
        json.JSONDecodeError,
        FileNotFoundError,
    ):
        return {}

    if not isinstance(loaded, dict):
        return {}

    users: dict[str, dict[str, object]] = {}

    for key, value in cast(
        dict[object, object],
        loaded,
    ).items():

        if (
            isinstance(key, str)
            and isinstance(value, dict)
        ):
            users[key] = cast(
                dict[str, object],
                value,
            )

    return users

def _write(
    self,
    users: dict[str, dict[str, object]],
) -> None:

    with self.filepath.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            users,
            f,
            indent=2,
            ensure_ascii=False,
        )

@staticmethod
def _hash_password(
    password: str,
    salt_hex: str,
) -> str:

    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        UserStore._HASH_ITERATIONS,
    )

    return digest.hex()

def register(
    self,
    username: str,
    password: str,
    confirm_password: str,
) -> User:

    cleaned_username = username.strip()

    if not cleaned_username:
        raise InvalidCredentialsError(
            "Username cannot be empty."
        )

    if len(cleaned_username) > 50:
        raise InvalidCredentialsError(
            "Username is too long."
        )

    if len(password) < self._MIN_PASSWORD_LENGTH:
        raise InvalidCredentialsError(
            "Password must be at least "
            f"{self._MIN_PASSWORD_LENGTH} characters long."
        )

    if password != confirm_password:
        raise InvalidCredentialsError(
            "Passwords do not match."
        )

    users = self._read()

    lookup_key = cleaned_username.lower()

    if lookup_key in users:
        raise UsernameTakenError(
            f'The username "{cleaned_username}" '
            "is already taken."
        )

    salt_hex = secrets.token_hex(16)

    password_hash = self._hash_password(
        password,
        salt_hex,
    )

    created_at = datetime.now().isoformat(
        timespec="seconds"
    )

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

def authenticate(
    self,
    username: str,
    password: str,
) -> User:

    users = self._read()

    record = users.get(
        username.strip().lower()
    )

    if record is None:
        raise InvalidCredentialsError(
            "Incorrect username or password."
        )

    stored_salt = record.get("salt")
    stored_hash = record.get("password_hash")

    if not isinstance(stored_salt, str):
        raise InvalidCredentialsError(
            "Incorrect username or password."
        )

    if not isinstance(stored_hash, str):
        raise InvalidCredentialsError(
            "Incorrect username or password."
        )

    attempted_hash = self._hash_password(
        password,
        stored_salt,
    )

    if not secrets.compare_digest(
        attempted_hash,
        stored_hash,
    ):
        raise InvalidCredentialsError(
            "Incorrect username or password."
        )

    return User(
        username=str(
            record.get(
                "username",
                username,
            )
        ),
        password_hash=stored_hash,
        salt=stored_salt,
        created_at=str(
            record.get(
                "created_at",
                "",
            )
        ),
    )

def get_user(
    self,
    username: str,
) -> User | None:

    users = self._read()

    record = users.get(
        username.strip().lower()
    )

    if not record:
        return None

    password_hash = record.get(
        "password_hash",
        "",
    )

    salt = record.get(
        "salt",
        "",
    )

    if not isinstance(password_hash, str):
        return None

    if not isinstance(salt, str):
        return None

    return User(
        username=str(
            record.get(
                "username",
                username,
            )
        ),
        password_hash=password_hash,
        salt=salt,
        created_at=str(
            record.get(
                "created_at",
                "",
            )
        ),
    )
```

# ---------------------------------------------------------------------------

# Search history

# ---------------------------------------------------------------------------

class SearchHistory:
def **init**(
self,
filepath: str = "search_history.json",
):
self.filepath = Path(filepath)

```
    if not self.filepath.exists():
        self._write([])

def _read(self) -> list[dict[str, object]]:

    try:
        with self.filepath.open(
            "r",
            encoding="utf-8",
        ) as f:
            loaded = json.load(f)

    except (
        json.JSONDecodeError,
        FileNotFoundError,
    ):
        return []

    if not isinstance(loaded, list):
        return []

    return [
        cast(dict[str, object], entry)
        for entry in cast(list[object], loaded)
        if isinstance(entry, dict)
    ]

def _write(
    self,
    entries: list[dict[str, object]],
) -> None:

    with self.filepath.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            entries,
            f,
            indent=2,
            ensure_ascii=False,
        )

def add(
    self,
    entry: dict[str, object],
) -> None:

    entries = self._read()

    entries.insert(
        0,
        entry,
    )

    self._write(
        entries[:100]
    )

def get_all(
    self,
) -> list[dict[str, object]]:
    return self._read()

def clear(self) -> None:
    self._write([])
```

# ---------------------------------------------------------------------------

# Suggestions

# ---------------------------------------------------------------------------

@lru_cache(maxsize=256)
def get_cached_suggestions(
partial_name: str,
) -> tuple[str, ...]:

```
try:
    return tuple(
        FDAClient().suggest_names(
            partial_name
        )
    )

except FDANetworkError:
    return ()
```

_SUGGESTION_PLACEHOLDER = "Did you mean…"

def render_search_suggestions(
session_state: dict[str, object],
drug_name_input: str,
) -> None:

```
cleaned = drug_name_input.strip()

if len(cleaned) < 2:
    return

suggestions = get_cached_suggestions(
    cleaned
)

suggestions = tuple(
    suggestion
    for suggestion in suggestions
    if suggestion.lower() != cleaned.lower()
)

if not suggestions:
    return

dropdown_key = (
    f"suggestion_select::{cleaned.lower()}"
)

options = [
    _SUGGESTION_PLACEHOLDER,
    *suggestions,
]

def apply_selected_suggestion() -> None:

    selected = cast(
        str,
        session_state.get(
            dropdown_key,
            _SUGGESTION_PLACEHOLDER,
        ),
    )

    if (
        selected
        and selected != _SUGGESTION_PLACEHOLDER
    ):
        session_state[
            "drug_name_input"
        ] = selected

        session_state[
            "trigger_search"
        ] = True

cast(
    Callable[..., object],
    getattr(
        cast(object, st),
        "selectbox",
    ),
)(
    "Did you mean:",
    options,
    key=dropdown_key,
    on_change=apply_selected_suggestion,
)
```

# ---------------------------------------------------------------------------

# UI helpers

# ---------------------------------------------------------------------------

def build_entry(
drug_name: str,
medication: Medication,
recalls: list[dict[str, str]],
simplified: dict[str, str],
) -> dict[str, object]:

```
return {
    "timestamp": datetime.now().isoformat(
        timespec="seconds"
    ),
    "query": drug_name,
    "generic_name": medication.generic_name,
    "brand_names": medication.brand_names,
    "recall_found": bool(recalls),
    "recall_count": len(recalls),
    "simplified": simplified,
}
```

def render_recall_banner(
recalls: list[dict[str, str]],
) -> None:

```
if recalls:

    cast(
        _StreamlitErrorAPI,
        cast(object, st),
    ).error(
        f"⚠️ {len(recalls)} recall notice(s) "
        "found for this medication."
    )

    for recall in recalls:

        title = (
            f"{recall['recalling_firm'] or 'Unknown firm'} "
            f"— {recall['report_date']} "
            f"(Class {recall['classification']})"
        )

        with cast(
            _StreamlitExpanderFactoryAPI,
            cast(object, st),
        ).expander(title):

            cast(
                _StreamlitWriteAPI,
                cast(object, st),
            ).write(
                "**Reason:** "
                f"{recall['reason_for_recall'] or 'Not specified'}"
            )

            cast(
                _StreamlitWriteAPI,
                cast(object, st),
            ).write(
                f"**Status:** {recall['status']}"
            )

            cast(
                _StreamlitWriteAPI,
                cast(object, st),
            ).write(
                f"**Product:** "
                f"{recall['product_description']}"
            )

else:

    cast(
        _StreamlitSuccessAPI,
        cast(object, st),
    ).success(
        "✅ No recent recalls found for this medication."
    )
```

# ---------------------------------------------------------------------------

# Profile sidebar

# ---------------------------------------------------------------------------

def render_profile_sidebar(
sidebar: object,
user_store: UserStore,
username: str,
) -> None:

```
sidebar_header = cast(
    Callable[[str], object],
    getattr(sidebar, "header"),
)

sidebar_write = cast(
    Callable[[str], object],
    getattr(sidebar, "write"),
)

sidebar_divider = cast(
    Callable[[], object],
    getattr(sidebar, "divider"),
)

sidebar_header("👤 Profile")

sidebar_write(
    f"**Username:** {username}"
)

user = user_store.get_user(username)

if user:
    sidebar_write(
        f"**Member since:** {user.created_at}"
    )

sidebar_write(
    "**Account status:** 🟢 Active"
)

sidebar_divider()
```

# ---------------------------------------------------------------------------

# Authentication UI

# ---------------------------------------------------------------------------

def render_auth_ui(
user_store: UserStore,
session_state: dict[str, object],
) -> None:

```
cast(
    Callable[[str], object],
    getattr(cast(object, st), "subheader"),
)(
    "Sign in to continue"
)

tabs = cast(
    _StreamlitTabsFactoryAPI,
    cast(object, st),
).tabs(
    [
        "Sign In",
        "Sign Up",
    ]
)

with tabs[0]:

    with cast(
        _StreamlitFormFactoryAPI,
        cast(object, st),
    ).form("sign_in_form"):

        username = cast(
            Callable[..., str],
            getattr(
                cast(object, st),
                "text_input",
            ),
        )(
            "Username",
            key="sign_in_username",
        )

        password = cast(
            Callable[..., str],
            getattr(
                cast(object, st),
                "text_input",
            ),
        )(
            "Password",
            type="password",
            key="sign_in_password",
        )

        submitted = cast(
            Callable[..., bool],
            getattr(
                cast(object, st),
                "form_submit_button",
            ),
        )(
            "Sign In"
        )

    if submitted:

        try:
            user = user_store.authenticate(
                username,
                password,
            )

        except InvalidCredentialsError as exc:

            cast(
                _StreamlitErrorAPI,
                cast(object, st),
            ).error(str(exc))

        else:

            session_state[
                "logged_in_user"
            ] = user.username

            cast(
                Callable[[], object],
                getattr(
                    cast(object, st),
                    "rerun",
                ),
            )()

with tabs[1]:

    with cast(
        _StreamlitFormFactoryAPI,
        cast(object, st),
    ).form("sign_up_form"):

        username = cast(
            Callable[..., str],
            getattr(
                cast(object, st),
                "text_input",
            ),
        )(
            "Choose a username",
            key="sign_up_username",
        )

        password = cast(
            Callable[..., str],
            getattr(
                cast(object, st),
                "text_input",
            ),
        )(
            "Choose a password",
            type="password",
            key="sign_up_password",
        )

        confirm = cast(
            Callable[..., str],
            getattr(
                cast(object, st),
                "text_input",
            ),
        )(
            "Confirm password",
            type="password",
            key="sign_up_confirm",
        )

        submitted = cast(
            Callable[..., bool],
            getattr(
                cast(object, st),
                "form_submit_button",
            ),
        )(
            "Sign Up"
        )

    if submitted:

        try:

            user = user_store.register(
                username,
                password,
                confirm,
            )

        except (
            InvalidCredentialsError,
            UsernameTakenError,
        ) as exc:

            cast(
                _StreamlitErrorAPI,
                cast(object, st),
            ).error(str(exc))

        else:

            session_state[
                "logged_in_user"
            ] = user.username

            cast(
                _StreamlitSuccessAPI,
                cast(object, st),
            ).success(
                f"Account created — welcome, "
                f"{user.username}!"
            )

            cast(
                Callable[[], object],
                getattr(
                    cast(object, st),
                    "rerun",
                ),
            )()
```

# ---------------------------------------------------------------------------

# AI Chat UI

# ---------------------------------------------------------------------------

def render_ai_chat(
medication: Medication | None,
api_key: str,
session_state: dict[str, object],
) -> None:

```
cast(
    Callable[[str], object],
    getattr(cast(object, st), "subheader"),
)(
    "🤖 Ask MedSimplify AI"
)

if medication is None:

    cast(
        Callable[[str], object],
        getattr(cast(object, st), "info"),
    )(
        "Search for a medication first to start "
        "a conversation with the AI."
    )

    return

medication_key = (
    medication.generic_name
    or medication.name
).lower()

chat_key = f"ai_chat::{medication_key}"

if (
    session_state.get(
        "active_chat_medication"
    )
    != medication_key
):

    session_state[
        "active_chat_medication"
    ] = medication_key

    session_state[
        chat_key
    ] = []

if chat_key not in session_state:
    session_state[chat_key] = []

messages = cast(
    list[dict[str, str]],
    session_state[chat_key],
)

cast(
    Callable[[str], object],
    getattr(cast(object, st), "caption"),
)(
    "Ask questions about "
    f"{medication.generic_name or medication.name}. "
    "The AI uses the available FDA label information "
    "as its medication context."
)

# Suggested questions
suggestion_cols = cast(
    list[object],
    getattr(cast(object, st), "columns"),
)(
    3
)

suggested_questions = [
    "What is this medicine used for?",
    "What are the important warnings?",
    "What side effects should I know about?",
]

for column, suggestion in zip(
    suggestion_cols,
    suggested_questions,
):

    with column:

        if cast(
            Callable[..., bool],
            getattr(column, "button"),
        )(
            suggestion,
            key=f"quick_question::{medication_key}::{suggestion}",
        ):

            session_state[
                "pending_ai_question"
            ] = suggestion

# Display existing conversation.
for message in messages:

    role = message.get(
        "role",
        "assistant",
    )

    content = message.get(
        "content",
        "",
    )

    with cast(
        Callable[..., object],
        getattr(
            cast(object, st),
            "chat_message",
        ),
    )(role):

        cast(
            Callable[[str], object],
            getattr(
                cast(object, st),
                "markdown",
            ),
        )(content)

if not api_key:

    cast(
        _StreamlitWarningAPI,
        cast(object, st),
    ).warning(
        "Add your Gemini API key in the sidebar "
        "to use MedSimplify AI."
    )

    return

question = session_state.pop(
    "pending_ai_question",
    None,
)

if not question:

    question = cast(
        Callable[..., str | None],
        getattr(
            cast(object, st),
            "chat_input",
        ),
    )(
        "Ask something about "
        f"{medication.generic_name or medication.name}..."
    )

if not question:
    return

question = str(question).strip()

if not question:
    return

messages.append(
    {
        "role": "user",
        "content": question,
    }
)

with cast(
    Callable[..., object],
    getattr(
        cast(object, st),
        "chat_message",
    ),
)("user"):

    cast(
        Callable[[str], object],
        getattr(
            cast(object, st),
            "markdown",
        ),
    )(question)

try:

    chatbot = AIChat(
        api_key=api_key
    )

    with cast(
        Callable[..., object],
        getattr(
            cast(object, st),
            "chat_message",
        ),
    )("assistant"):

        with cast(
            _StreamlitSpinnerFactoryAPI,
            cast(object, st),
        ).spinner(
            "Thinking..."
        ):

            answer = chatbot.ask(
                medication=medication,
                question=question,
                conversation=messages[:-1],
            )

        cast(
            Callable[[str], object],
            getattr(
                cast(object, st),
                "markdown",
            ),
        )(answer)

    messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

except AITranslationError as exc:

    cast(
        _StreamlitErrorAPI,
        cast(object, st),
    ).error(
        f"AI chat could not respond: {exc}"
    )
```

# ---------------------------------------------------------------------------

# Main

# ---------------------------------------------------------------------------

def main() -> None:

```
cast(
    _StreamlitPageConfigAPI,
    cast(object, st),
).set_page_config(
    page_title="MedSimplify",
    page_icon="💊",
    layout="centered",
)

cast(
    Callable[[str], object],
    getattr(cast(object, st), "title"),
)(
    "💊 MedSimplify"
)

cast(
    Callable[[str], object],
    getattr(cast(object, st), "caption"),
)(
    "Understand your medication in plain language."
)

session_state = cast(
    dict[str, object],
    getattr(
        cast(object, st),
        "session_state",
    ),
)

if "logged_in_user" not in session_state:
    session_state["logged_in_user"] = None

user_store = UserStore()

# -----------------------------------------------------------------------
# Authentication
# -----------------------------------------------------------------------

if not session_state.get(
    "logged_in_user"
):

    render_auth_ui(
        user_store,
        session_state,
    )

    return

logged_in_user = cast(
    str,
    session_state["logged_in_user"],
)

# -----------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------

sidebar = getattr(
    cast(object, st),
    "sidebar",
)

render_profile_sidebar(
    sidebar,
    user_store,
    logged_in_user,
)

safe_username = re.sub(
    r"[^a-z0-9_-]",
    "_",
    logged_in_user.lower(),
)

history = SearchHistory(
    filepath=(
        f"search_history_{safe_username}.json"
    )
)

api_key = ""

with sidebar:

    cast(
        Callable[[str], object],
        getattr(
            cast(object, sidebar),
            "write",
        ),
    )(
        f"👤 Signed in as "
        f"**{logged_in_user}**"
    )

    if cast(
        Callable[[str], bool],
        getattr(
            cast(object, sidebar),
            "button",
        ),
    )(
        "Log out"
    ):

        session_state[
            "logged_in_user"
        ] = None

        cast(
            Callable[[], object],
            getattr(
                cast(object, st),
                "rerun",
            ),
        )()

    cast(
        Callable[[], object],
        getattr(
            cast(object, sidebar),
            "divider",
        ),
    )()

    cast(
        Callable[[str], object],
        getattr(
            cast(object, sidebar),
            "header",
        ),
    )(
        "⚙️ Settings"
    )

    api_key = cast(
        Callable[..., str],
        getattr(
            cast(object, sidebar),
            "text_input",
        ),
    )(
        "Gemini API Key",
        value=os.environ.get(
            "GEMINI_API_KEY",
            "",
        ),
        type="password",
        help=(
            "Your Gemini API key is used for "
            "AI features during this session."
        ),
    )

    cast(
        Callable[[], object],
        getattr(
            cast(object, sidebar),
            "divider",
        ),
    )()

    cast(
        Callable[[str], object],
        getattr(
            cast(object, sidebar),
            "header",
        ),
    )(
        "📚 Search History"
    )

    entries = history.get_all()

    if entries:

        for entry in entries[:15]:

            flag = (
                "⚠️"
                if entry.get("recall_found")
                else ""
            )

            cast(
                Callable[[str], object],
                getattr(
                    cast(object, sidebar),
                    "write",
                ),
            )(
                f"{flag} **{entry.get('query', '')}** "
                f"— {entry.get('timestamp', '')}"
            )

        if cast(
            Callable[[str], bool],
            getattr(
                cast(object, sidebar),
                "button",
            ),
        )(
            "Clear history"
        ):

            history.clear()

            cast(
                Callable[[], object],
                getattr(
                    cast(object, st),
                    "rerun",
                ),
            )()

    else:

        cast(
            Callable[[str], object],
            getattr(
                cast(object, sidebar),
                "write",
            ),
        )(
            "No searches yet."
        )

# -----------------------------------------------------------------------
# Search input
# -----------------------------------------------------------------------

if "drug_name_input" not in session_state:
    session_state["drug_name_input"] = ""

if "trigger_search" not in session_state:
    session_state["trigger_search"] = False

drug_name_input = cast(
    Callable[..., str],
    getattr(
        cast(object, st),
        "text_input",
    ),
)(
    "Enter a medication name",
    placeholder="e.g. ibuprofen",
    key="drug_name_input",
)

render_search_suggestions(
    session_state,
    drug_name_input,
)

search_clicked = cast(
    Callable[..., bool],
    getattr(
        cast(object, st),
        "button",
    ),
)(
    "🔎 Search",
    type="primary",
)

if not (
    search_clicked
    or session_state.get("trigger_search")
):
    return

session_state[
    "trigger_search"
] = False

# -----------------------------------------------------------------------
# Validate
# -----------------------------------------------------------------------

try:

    drug_name = Medication.validate_name(
        drug_name_input
    )

except InvalidMedicationNameError as exc:

    cast(
        _StreamlitErrorAPI,
        cast(object, st),
    ).error(str(exc))

    return

fda_client = FDAClient()

medication: Medication | None = None
recalls: list[dict[str, str]] = []

# -----------------------------------------------------------------------
# FDA lookup
# -----------------------------------------------------------------------

with cast(
    _StreamlitSpinnerFactoryAPI,
    cast(object, st),
).spinner(
    f"Looking up '{drug_name}'..."
):

    try:

        medication = fda_client.fetch_label(
            drug_name
        )

    except MedicationNotFoundError as exc:

        cast(
            _StreamlitWarningAPI,
            cast(object, st),
        ).warning(str(exc))

        return

    except FDANetworkError as exc:

        cast(
            _StreamlitErrorAPI,
            cast(object, st),
        ).error(
            f"Network problem while fetching "
            f"drug info: {exc}"
        )

        return

    missing = medication.has_missing_fields()

    if missing:

        cast(
            Callable[[str], object],
            getattr(
                cast(object, st),
                "info",
            ),
        )(
            "Note: the FDA label was missing data for: "
            + ", ".join(missing)
        )

    try:

        recalls = fda_client.fetch_recalls(
            medication.generic_name
            or drug_name
        )

    except FDANetworkError as exc:

        cast(
            _StreamlitWarningAPI,
            cast(object, st),
        ).warning(
            f"Could not check recalls right now: {exc}"
        )

        recalls = []

# -----------------------------------------------------------------------
# Medication heading
# -----------------------------------------------------------------------

cast(
    Callable[[str], object],
    getattr(
        cast(object, st),
        "subheader",
    ),
)(
    (
        medication.generic_name
        or drug_name
    ).title()
)

if medication.brand_names:

    cast(
        Callable[[str], object],
        getattr(
            cast(object, st),
            "caption",
        ),
    )(
        "Brand names: "
        + ", ".join(
            medication.brand_names
        )
    )

# -----------------------------------------------------------------------
# Image
# -----------------------------------------------------------------------

try:

    image_url = ImageClient().fetch_thumbnail_url(
        medication,
        drug_name,
    )

    cast(
        Callable[..., object],
        getattr(
            cast(object, st),
            "image",
        ),
    )(
        image_url,
        caption=(
            f"Image of "
            f"{medication.generic_name or drug_name}"
        ),
        width=300,
    )

except ImageNotFoundError:

    cast(
        Callable[[str], object],
        getattr(
            cast(object, st),
            "caption",
        ),
    )(
        "No reference image available "
        "for this medication."
    )

# -----------------------------------------------------------------------
# Recalls
# -----------------------------------------------------------------------

render_recall_banner(recalls)

# -----------------------------------------------------------------------
# Warning keywords
# -----------------------------------------------------------------------

keywords = medication.extract_warning_keywords()

if keywords:

    cast(
        Callable[[str], object],
        getattr(
            cast(object, st),
            "markdown",
        ),
    )(
        "**Key warning phrases:**"
    )

    cast(
        _StreamlitWriteAPI,
        cast(object, st),
    ).write(
        " • " + "\n • ".join(keywords)
    )

# -----------------------------------------------------------------------
# AI simplified information
# -----------------------------------------------------------------------

cast(
    Callable[[str], object],
    getattr(
        cast(object, st),
        "subheader",
    ),
)(
    "📖 Plain-Language Information"
)

simplified: dict[str, str] = {}

sections = {
    "Usage": medication.usage,
    "Dosage & Administration": medication.dosage,
    "Warnings": medication.warnings,
    "Side Effects": medication.side_effects,
}

markdown_fn = cast(
    Callable[[str], object],
    getattr(
        cast(object, st),
        "markdown",
    ),
)

write_fn = cast(
    _StreamlitWriteAPI,
    cast(object, st),
)

if not api_key:

    cast(
        _StreamlitWarningAPI,
        cast(object, st),
    ).warning(
        "No Gemini API key provided — "
        "showing the original FDA text."
    )

    for label, text in sections.items():

        markdown_fn(
            f"### {label}"
        )

        write_fn.write(
            text or "_No data available._"
        )

else:

    translator = AITranslator(
        api_key=api_key
    )

    for label, text in sections.items():

        markdown_fn(
            f"### {label}"
        )

        try:

            with cast(
                _StreamlitSpinnerFactoryAPI,
                cast(object, st),
            ).spinner(
                f"Simplifying '{label}'..."
            ):

                simple_text = translator.simplify(
                    text,
                    label,
                )

            write_fn.write(
                simple_text
            )

            simplified[label] = simple_text

        except AITranslationError as exc:

            cast(
                _StreamlitErrorAPI,
                cast(object, st),
            ).error(
                f"Could not simplify this section: "
                f"{exc}"
            )

            write_fn.write(
                text or "_No data available._"
            )

        time.sleep(0.2)

# -----------------------------------------------------------------------
# AI Chat
# -----------------------------------------------------------------------

render_ai_chat(
    medication,
    api_key,
    session_state,
)

# -----------------------------------------------------------------------
# Save search
# -----------------------------------------------------------------------

entry = build_entry(
    drug_name,
    medication,
    recalls,
    simplified,
)

history.add(entry)

cast(
    Callable[[str], object],
    getattr(
        cast(object, st),
        "toast",
    ),
)(
    "✅ Search saved to history."
)
```

# ---------------------------------------------------------------------------

# Run

# ---------------------------------------------------------------------------

if **name** == "**main**":
main()
