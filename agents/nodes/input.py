import re
import unicodedata

from agents.state import AgentState, AgentStateUpdate


EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_PATTERN = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[ .-]?)?(?:\(?\d{2,4}\)?[ .-]?){2}\d{3,4}(?!\w)"
)
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CARD_PATTERN = re.compile(r"\b(?:\d[ -]?){13,19}\b")
IP_ADDRESS_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
HTML_TAG_PATTERN = re.compile(r"<[^>]*>")
CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
UNSUPPORTED_SYMBOL_PATTERN = re.compile(r"[^\w\s.,!?;:'\"()\[\]{}\-_/+*=#%]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_prompt(prompt: str) -> str:
    """Return a sanitized, router-ready version of a user prompt."""
    normalized = unicodedata.normalize("NFKC", prompt)
    normalized = CONTROL_CHARACTER_PATTERN.sub(" ", normalized)
    normalized = HTML_TAG_PATTERN.sub(" ", normalized)
    normalized = EMAIL_PATTERN.sub("[email removed]", normalized)
    normalized = SSN_PATTERN.sub("[ssn removed]", normalized)
    normalized = CARD_PATTERN.sub("[payment card removed]", normalized)
    normalized = PHONE_PATTERN.sub("[phone removed]", normalized)
    normalized = IP_ADDRESS_PATTERN.sub("[ip address removed]", normalized)
    normalized = UNSUPPORTED_SYMBOL_PATTERN.sub(" ", normalized)
    return WHITESPACE_PATTERN.sub(" ", normalized).strip()


def make_normalize_input_node():
    def normalize_input_node(state: AgentState) -> AgentStateUpdate:
        return {"prompt": normalize_prompt(state.get("prompt", ""))}

    return normalize_input_node
