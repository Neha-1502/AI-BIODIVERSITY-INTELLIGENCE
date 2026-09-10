"""
test_conversation_flow.py

Standalone sanity check for conversation.py's slot-filling + clarifying-question
flow, run without spinning up Streamlit. Covers:

  1. A genuinely vague/incomplete message -- confirms extraction doesn't
     hallucinate values it wasn't given, and that the clarifying question
     reads naturally (brief's "Conversational Intelligence" criterion).
  2. A message with qualitative-but-inferable values ("very low rainfall") --
     confirms the LLM extractor converts vague language to representative
     numbers per the rules in conversation.py's EXTRACTION_SYSTEM_PROMPT.
  3. A fully-specified message -- confirms the conversation reaches
     "ready for recommendation" in a single turn, matching the brief's
     own example conversation.

Run with:
    python scripts/test_conversation_flow.py

Requires a working LLM API key in .env (same as reasoning.py/app.py).
"""

import conversation


def run_case(label: str, message: str):
    print("\n" + "=" * 70)
    print(f"CASE: {label}")
    print("=" * 70)
    print(f"User message: {message!r}")

    state = conversation.ConversationState()
    state.add_user_message(message)

    extracted = conversation.extract_slots_from_message(message)
    print(f"Extracted: {extracted}")

    state.merge_extracted_values(extracted)
    print(f"State after merge: {state.collected}")

    missing = state.missing_required_slots()
    if missing:
        question = conversation.generate_clarifying_question(missing)
        print(f"Still missing: {missing}")
        print(f"Clarifying question: {question}")
    else:
        print("Ready for recommendation -- no clarifying question needed.")

    return state


if __name__ == "__main__":
    # Case 1: genuinely vague, no numbers, no land-use hint at all.
    # Expect: little to nothing extracted, a natural clarifying question
    # asking for all 3 required slots (matches the brief's own example:
    # "Biodiversity is declining on my land" -> asks for SOC/rainfall/land use).
    run_case(
        "Vague, no numbers",
        "My soil feels dry and nothing grows well anymore.",
    )

    # Case 2: qualitative language that should still map to representative
    # numbers per the extraction prompt's rules, but land use is still missing.
    run_case(
        "Qualitative values, land use missing",
        "It's very degraded soil here and we get very low rainfall most years.",
    )

    # Case 3: fully specified in one message -- should need zero follow-up.
    run_case(
        "Fully specified in one turn",
        "My soil organic carbon is around 0.3%, we get very low rainfall, "
        "and it's monoculture wheat.",
    )

    # Case 4: multi-turn -- partial info first, then the rest in a follow-up.
    print("\n" + "=" * 70)
    print("CASE: Multi-turn (partial, then follow-up)")
    print("=" * 70)
    state = conversation.ConversationState()

    msg1 = "Biodiversity is declining on my land."
    state.add_user_message(msg1)
    extracted1 = conversation.extract_slots_from_message(msg1)
    state.merge_extracted_values(extracted1)
    missing1 = state.missing_required_slots()
    print(f"Turn 1 -- message: {msg1!r}")
    print(f"Turn 1 -- extracted: {extracted1}, still missing: {missing1}")
    if missing1:
        print(f"Turn 1 -- clarifying question: {conversation.generate_clarifying_question(missing1)}")

    msg2 = "Sure -- SOC is about 0.4%, rainfall is roughly 280mm/year, and it's monoculture wheat."
    state.add_user_message(msg2)
    extracted2 = conversation.extract_slots_from_message(msg2)
    state.merge_extracted_values(extracted2)
    missing2 = state.missing_required_slots()
    print(f"Turn 2 -- message: {msg2!r}")
    print(f"Turn 2 -- extracted: {extracted2}, still missing: {missing2}")
    print(f"Turn 2 -- final collected state: {state.collected}")
    print(f"Ready for recommendation: {state.is_ready_for_recommendation()}")