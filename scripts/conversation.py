"""
conversation.py

Phase 3 - Conversational Intelligence layer.

Handles:
  - Tracking what environmental variables the user has provided so far
    (multi-turn memory)
  - Detecting which required variables are still missing
  - Using the LLM to extract structured values from free-text user messages
    (so "it's pretty dry here, maybe 250mm a year" gets parsed into
    {"rainfall": 250})
  - Generating a natural clarifying question when inputs are incomplete

This directly satisfies the brief's "Conversational Intelligence" requirement:
multi-turn memory + clarifying questions + context adaptation.
"""

import json
import llm_client

# The minimum set of variables needed before we can generate a grounded
# recommendation. Expand this list as you add more metrics/interventions to
# structured_knowledge.json.
REQUIRED_SLOTS = [
    "soil_organic_carbon",   # % , e.g. 0.3
    "rainfall",               # mm/year, e.g. 250
    "land_use_type",          # one of: monoculture, intercropping, agroforestry,
                               #         natural_forest, grassland, urban, fallow
]

# Optional slots -- nice to have for richer reasoning, but not required to
# generate a first recommendation.
OPTIONAL_SLOTS = [
    "soil_ph",
    "soil_moisture",
    "temperature",
    "habitat_fragmentation_index",
    "pollution_index",
    "region",
]

ALL_SLOTS = REQUIRED_SLOTS + OPTIONAL_SLOTS


class ConversationState:
    """Holds everything collected so far in this session."""

    def __init__(self):
        self.collected: dict = {}
        self.history: list[dict] = []   # [{"role": "user"/"assistant", "content": "..."}]

    def missing_required_slots(self) -> list[str]:
        return [slot for slot in REQUIRED_SLOTS if slot not in self.collected]

    def is_ready_for_recommendation(self) -> bool:
        return len(self.missing_required_slots()) == 0

    def add_user_message(self, text: str):
        self.history.append({"role": "user", "content": text})

    def add_assistant_message(self, text: str):
        self.history.append({"role": "assistant", "content": text})

    def merge_extracted_values(self, extracted: dict):
        """Only accepts known slot names, ignores anything the LLM hallucinated
        as a key that isn't in ALL_SLOTS -- keeps state clean."""
        for key, value in extracted.items():
            if key in ALL_SLOTS and value is not None:
                self.collected[key] = value


EXTRACTION_SYSTEM_PROMPT = f"""You extract structured environmental metric values from
a user's message in a conversation about land/biodiversity conditions.

Known slot names you should extract, if mentioned or reasonably inferable:
{json.dumps(ALL_SLOTS)}

Rules:
- Only extract a value if the user's message actually states or clearly implies it.
- Do NOT guess or invent values that were not communicated.
- "land_use_type" must be one of: monoculture, intercropping, agroforestry,
  natural_forest, grassland, urban, fallow -- map the user's description to the
  closest one of these, or omit the key if it doesn't fit any of them.
- Numeric values (soil_organic_carbon, rainfall, soil_ph, soil_moisture,
  temperature, habitat_fragmentation_index, pollution_index) should be plain
  numbers, not strings, and not ranges (pick the midpoint of a stated range).
- Vague qualitative statements like "low rainfall" or "very dry soil" without a
  number should be converted to a reasonable representative number for that
  category (e.g. "low rainfall" -> 250, "very degraded soil" for SOC -> 0.3)
  ONLY if the user does not give an exact figure -- prefer exact figures when given.
- Output ONLY a flat JSON object of extracted slot_name: value pairs. Omit any
  slot not mentioned. If nothing can be extracted, output {{}}.
"""


def extract_slots_from_message(user_message: str) -> dict:
    """Calls the LLM to pull structured values out of a free-text message."""
    try:
        return llm_client.chat_json(
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_prompt=f"User message: \"{user_message}\"\n\nExtract the slot values as JSON.",
        )
    except (json.JSONDecodeError, Exception) as e:
        print(f"[warning] slot extraction failed: {e}")
        return {}


def generate_clarifying_question(missing_slots: list[str]) -> str:
    """Turns a list of missing required slots into one natural question,
    matching the brief's example: 'Can you provide soil organic carbon %,
    rainfall pattern, and land use type?'"""
    friendly_names = {
        "soil_organic_carbon": "soil organic carbon % (or a general sense of soil health)",
        "rainfall": "rainfall pattern (mm/year, or just 'low/moderate/high')",
        "land_use_type": "current land use (e.g. monoculture crop, agroforestry, grassland)",
        "soil_ph": "soil pH",
        "soil_moisture": "soil moisture level",
        "temperature": "average temperature",
        "habitat_fragmentation_index": "how fragmented/connected the surrounding habitat is",
        "pollution_index": "any known pollution sources nearby (runoff, chemicals)",
        "region": "your region or climate zone",
    }
    items = [friendly_names.get(s, s) for s in missing_slots]

    if len(items) == 1:
        joined = items[0]
    elif len(items) == 2:
        joined = f"{items[0]} and {items[1]}"
    else:
        joined = ", ".join(items[:-1]) + f", and {items[-1]}"

    return f"To give you a grounded recommendation, could you share: {joined}?"


if __name__ == "__main__":
    # Sanity check with the brief's own example conversation
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