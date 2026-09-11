"""
conversation.py

Conversational intelligence layer.

Handles:
  - Multi-turn memory of environmental variables
  - Extraction of structured values from free text (with prior-turn context)
  - Clarifying questions when required inputs are incomplete
  - Follow-up prompts when the user implies an extra metric without a value
"""

import json
import re
import llm_client

# Minimum variables before a grounded recommendation (brief example: SOC, rainfall, land use).
REQUIRED_SLOTS = [
    "soil_organic_carbon",
    "rainfall",
    "land_use_type",
]

OPTIONAL_SLOTS = [
    "soil_ph",
    "soil_moisture",
    "temperature",
    "habitat_fragmentation_index",
    "habitat_diversity",
    "species_richness",
    "pollution_index",
    "deforestation_rate",
    "region",
]

ALL_SLOTS = REQUIRED_SLOTS + OPTIONAL_SLOTS

FRIENDLY_NAMES = {
    "soil_organic_carbon": "soil organic carbon %",
    "rainfall": "rainfall pattern",
    "land_use_type": "land use type",
    "soil_ph": "soil pH",
    "soil_moisture": "soil moisture level (% volumetric, or dry/adequate)",
    "temperature": "mean annual temperature (°C)",
    "habitat_fragmentation_index": "how fragmented the surrounding habitat is (0 contiguous – 1 fully fragmented)",
    "habitat_diversity": "habitat diversity (0 single cover type – 1 mixed mosaic)",
    "species_richness": "species richness relative to local potential (0–100, or low/moderate/high)",
    "pollution_index": "pollution / chemical pressure nearby (0–100, or low/moderate/high)",
    "deforestation_rate": "rate of forest/tree cover loss (% per year)",
    "region": "your region or climate zone",
}

# If the user asks about a topic we do not yet have a number for, ask before guessing.
_IMPLIED_SLOT_PATTERNS = [
    (r"\bpH\b|acidic|alkaline|lime", "soil_ph"),
    (r"moist|dry soil|waterlog", "soil_moisture"),
    (r"temperat|\bheat\b|hot climate", "temperature"),
    (r"fragment|corridor|isolated patch", "habitat_fragmentation_index"),
    (r"habitat divers|mosaic|cover type", "habitat_diversity"),
    (r"species rich|few species|biodiversity indicator", "species_richness"),
    (r"pollut|pesticide|runoff|chemical", "pollution_index"),
    (r"deforest|tree cover loss|clear.?cut|logging", "deforestation_rate"),
]


class ConversationState:
    """Holds everything collected so far in this session."""

    def __init__(self):
        self.collected: dict = {}
        self.history: list[dict] = []
        self.last_focus: str | None = None
        self.asked_optional: set[str] = set()

    def missing_required_slots(self) -> list[str]:
        return [slot for slot in REQUIRED_SLOTS if slot not in self.collected]

    def is_ready_for_recommendation(self) -> bool:
        return len(self.missing_required_slots()) == 0

    def add_user_message(self, text: str):
        self.history.append({"role": "user", "content": text})

    def add_assistant_message(self, text: str):
        self.history.append({"role": "assistant", "content": text})

    def merge_extracted_values(self, extracted: dict):
        for key, value in extracted.items():
            if key in ALL_SLOTS and value is not None:
                self.collected[key] = value


EXTRACTION_SYSTEM_PROMPT = f"""You extract structured environmental metric values from
a user's message in a conversation about land/biodiversity conditions.

Known slot names you should extract, if mentioned or reasonably inferable:
{json.dumps(ALL_SLOTS)}

Rules:
- Use the prior conversation only to disambiguate the latest user message.
- Only extract a value if the latest user message states or clearly implies it.
- Do NOT guess or invent values that were not communicated.
- "land_use_type" must be one of: monoculture, intercropping, agroforestry,
  natural_forest, grassland, urban, fallow -- map the user's description to the
  closest one of these, or omit the key if it doesn't fit any of them.
- Numeric values (soil_organic_carbon, rainfall, soil_ph, soil_moisture,
  temperature, habitat_fragmentation_index, habitat_diversity, species_richness,
  pollution_index, deforestation_rate) should be plain numbers, not strings.
  For a stated range, pick the midpoint.
- Vague qualitative statements without a number should be converted to a
  representative number ONLY if the user does not give an exact figure:
    rainfall: low=250, moderate=700, high=1200
    soil_organic_carbon: very degraded=0.3, poor=0.7, moderate=1.5
    soil_moisture: dry/arid=6, adequate=18
    temperature: hot=32, temperate=18
    pollution: low=15, moderate=45, high=75
    habitat_fragmentation: low=0.2, high=0.7
    habitat_diversity: low=0.25, high=0.75
    species_richness: low=25, moderate=55, high=80
    deforestation: noticeable/high=2.5
- Output ONLY a flat JSON object of extracted slot_name: value pairs. Omit any
  slot not mentioned. If nothing can be extracted, output {{}}.
"""


def extract_slots_from_message(user_message: str, history: list | None = None) -> dict:
    """Pull structured values out of free text, using recent turns as context."""
    history_block = ""
    if history:
        recent = history[-8:]
        rendered = []
        for turn in recent:
            content = str(turn.get("content", ""))[:500]
            rendered.append(f"{turn.get('role', 'user')}: {content}")
        history_block = "Prior conversation:\n" + "\n".join(rendered) + "\n\n"

    try:
        return llm_client.chat_json(
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_prompt=(
                f"{history_block}Latest user message: \"{user_message}\"\n\n"
                "Extract newly stated slot values as JSON."
            ),
        )
    except llm_client.LLMUnavailableError:
        raise
    except (json.JSONDecodeError, Exception) as e:
        print(f"[warning] slot extraction failed: {e}")
        return {}


def _join_friendly(slots: list[str]) -> str:
    items = [FRIENDLY_NAMES.get(s, s) for s in slots]
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def generate_clarifying_question(missing_slots: list[str], collected: dict | None = None) -> str:
    """Natural clarifying question; mentions what is already known when useful."""
    if set(missing_slots) == set(REQUIRED_SLOTS):
        return "Can you provide soil organic carbon %, rainfall pattern, and land use type?"
    joined = _join_friendly(missing_slots)
    if collected:
        known_bits = []
        for key, value in collected.items():
            label = key.replace("_", " ")
            known_bits.append(f"{label}={value}")
        known = "; ".join(known_bits[:6])
        return f"I already have {known}. Can you provide {joined}?"
    return f"Can you provide {joined}?"


def implied_optional_slots(user_message: str, collected: dict, already_asked: set | None = None) -> list[str]:
    """Slots the user is asking about that we still lack a value for."""
    asked = already_asked or set()
    text = user_message.lower()
    found = []
    for pattern, slot in _IMPLIED_SLOT_PATTERNS:
        if slot in collected or slot in asked or slot in found:
            continue
        if re.search(pattern, text, flags=re.IGNORECASE):
            found.append(slot)
    return found


if __name__ == "__main__":
    state = ConversationState()
    state.add_user_message("Biodiversity is declining on my land")

    print("Missing slots:", state.missing_required_slots())
    print("Clarifying question:", generate_clarifying_question(state.missing_required_slots()))

    print("\n--- Testing extraction (requires a working LLM API key) ---")
    test_message = "My soil organic carbon is around 0.3%, we get very low rainfall, and it's monoculture wheat"
    extracted = extract_slots_from_message(test_message)
    print("Extracted:", extracted)
    state.merge_extracted_values(extracted)
    print("State after merge:", state.collected)
    print("Still missing:", state.missing_required_slots())
    print("Ready for recommendation:", state.is_ready_for_recommendation())
