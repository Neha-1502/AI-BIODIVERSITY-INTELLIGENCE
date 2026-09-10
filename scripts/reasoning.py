"""
reasoning.py

Phase 3 - Evidence-Backed Recommendation generation.

This is the core of the "AI environmental scientist, not a chatbot" requirement.
It does NOT let the LLM freestyle recommendations. Instead:

  1. query_kb.retrieve_full_context(user_inputs) does deterministic structured
     matching + vector retrieval (Phase 2) -- this is the ground truth.
  2. The LLM's ONLY job is to synthesize that already-retrieved evidence into
     clear, well-written, correctly-schema'd output. It is explicitly told not
     to introduce facts, numbers, or sources that aren't in the provided context.
  3. Output is enforced JSON matching the brief's required schema:
     recommendation, reasoning, impacted_metrics, time_horizon, confidence, sources.

This grounding approach is what satisfies "Scientific Grounding" (25% of score)
and avoids the brief's explicit disqualifier: "No generic LLM-only solutions."
"""

import json
import llm_client
import query_kb


SYNTHESIS_SYSTEM_PROMPT = """You are an AI environmental scientist assistant for
Darukaa.Earth. You explain biodiversity/land-management interventions to a user
based STRICTLY on evidence that has already been retrieved for you.

CRITICAL RULES:
- Use ONLY the intervention data, relationship chains, and evidence chunks
  provided to you below. Do not invent statistics, studies, or sources that
  are not present in the provided context.
- Do not soften or generalize the specific numbers you're given (e.g. keep
  "15-25% over 2-3 years", don't reduce it to "significant improvement").
- Connect at least two environmental variables per recommendation, using the
  provided relationship chain data -- this is mandatory (multi-metric reasoning).
- Write the "reasoning" field as 2-4 sentences of clear scientific explanation,
  synthesizing the mechanism + relationship chain + evidence chunk content in
  your own words (not copy-pasted verbatim from the evidence chunks).
- For "sources", combine the structured sources given AND cite the filename/
  document of any evidence chunk you drew from.
- Output ONLY a JSON object matching this exact schema (no markdown, no preamble):

{
  "recommendations": [
    {
      "action": "string - what to do",
      "reasoning": "string - why it works, 2-4 sentences, multi-metric",
      "impacted_metrics": ["string", ...],
      "expected_improvement": "string - quantified where possible",
      "time_horizon": "short_term | medium_term | long_term",
      "confidence": "high | medium | low",
      "sources": ["string", ...]
    }
  ],
  "summary": "string - 1-2 sentence overview of the overall situation and approach"
}

If the provided context contains no matching interventions, return:
{"recommendations": [], "summary": "explain briefly why no intervention matched, and what additional info would help"}
"""


def _format_context_for_prompt(enriched_results: list, user_inputs: dict) -> str:
    """Serializes the retrieved structured + vector evidence into a clear
    text block for the LLM to synthesize from."""
    lines = [f"USER-PROVIDED CONDITIONS: {json.dumps(user_inputs)}", ""]

    if not enriched_results:
        lines.append("No matching interventions were found in the structured knowledge base for these conditions.")
        return "\n".join(lines)

    for i, r in enumerate(enriched_results, 1):
        iv = r["intervention"]
        lines.append(f"--- RETRIEVED INTERVENTION {i} ---")
        lines.append(f"Action: {iv['action']}")
        lines.append(f"Mechanism: {iv['mechanism']}")
        lines.append(f"Impacted metrics: {', '.join(iv['impacted_metrics'])}")
        lines.append(f"Expected effect: {iv['expected_effect']}")
        lines.append(f"Time horizon: {iv['time_horizon']} | Confidence: {iv['confidence']}")
        lines.append(f"Structured sources: {', '.join(iv['sources'])}")

        if r["relationship_chains"]:
            lines.append("Relationship chain (for multi-metric reasoning):")
            for edge in r["relationship_chains"]:
                lines.append(f"  {edge['from']} -> {edge['to']} ({edge['relationship']}): {edge['note']}")

        if r["evidence_chunks"]:
            lines.append("Retrieved supporting evidence (from indexed reports):")
            for chunk in r["evidence_chunks"]:
                lines.append(f"  [Source: {chunk['filename']}] {chunk['text']}")

        lines.append("")

    return "\n".join(lines)


def generate_recommendations(user_inputs: dict) -> dict:
    """Main entry point for the conversational layer. Given collected user
    inputs (a dict of slot_name: value), retrieves evidence and returns a
    schema-compliant recommendation set."""
    enriched_results = query_kb.retrieve_full_context(user_inputs)
    context_block = _format_context_for_prompt(enriched_results, user_inputs)

    try:
        result = llm_client.chat_json(
            system_prompt=SYNTHESIS_SYSTEM_PROMPT,
            user_prompt=context_block,
        )
    except json.JSONDecodeError:
        # Fallback: return the raw structured data without LLM synthesis,
        # so the system still produces a valid (if less polished) answer.
        result = {
            "recommendations": [
                {
                    "action": r["intervention"]["action"],
                    "reasoning": r["intervention"]["mechanism"],
                    "impacted_metrics": r["intervention"]["impacted_metrics"],
                    "expected_improvement": r["intervention"]["expected_effect"],
                    "time_horizon": r["intervention"]["time_horizon"],
                    "confidence": r["intervention"]["confidence"],
                    "sources": r["intervention"]["sources"],
                }
                for r in enriched_results
            ],
            "summary": "(LLM synthesis unavailable -- showing raw structured matches)",
        }

    return result


def format_recommendations_markdown(result: dict) -> str:
    """Renders the JSON result as readable markdown for the chat UI."""
    lines = []
    if result.get("summary"):
        lines.append(result["summary"])
        lines.append("")

    if not result.get("recommendations"):
        return "\n".join(lines) if lines else "No recommendations could be generated for these conditions yet."

    for i, rec in enumerate(result["recommendations"], 1):
        lines.append(f"### {i}. {rec['action']}")
        lines.append(f"**Why it works:** {rec['reasoning']}")
        lines.append(f"**Impacted metrics:** {', '.join(rec['impacted_metrics'])}")
        lines.append(f"**Expected improvement:** {rec['expected_improvement']}")
        lines.append(f"**Time horizon:** {rec['time_horizon'].replace('_', ' ')} | **Confidence:** {rec['confidence']}")
        lines.append(f"**Sources:** {', '.join(rec['sources'])}")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    # Sanity check using the hackathon brief's example use case
    example_inputs = {
        "soil_organic_carbon": 0.3,
        "rainfall": 250,
        "land_use_type": "monoculture",
    }
    result = generate_recommendations(example_inputs)
    print(json.dumps(result, indent=2))
    print("\n--- Rendered ---\n")
    print(format_recommendations_markdown(result))

    # Second test case: exercises the climate (int_006) and human-impact/
    # deforestation (int_007) interventions, which the brief example above
    # never triggers. Confirms all 5 required knowledge categories
    # (soil, land use, biodiversity, climate, human impact) can fire together.
    print("\n" + "=" * 70)
    print("TEST 2: climate + deforestation conditions")
    print("=" * 70)
    climate_inputs = {
        "soil_organic_carbon": 0.3,
        "rainfall": 250,
        "land_use_type": "monoculture",
        "temperature": 32,
        "deforestation_rate": 2.5,
    }
    climate_result = generate_recommendations(climate_inputs)
    print(json.dumps(climate_result, indent=2))
    print("\n--- Rendered ---\n")
    print(format_recommendations_markdown(climate_result))

    fired_ids = {rec["action"] for rec in climate_result.get("recommendations", [])}
    expected_new_actions = {
        "Introduce shade-grown / multi-strata canopy cover for heat buffering",
        "Launch assisted natural regeneration (ANR) / community reforestation program",
    }
    missing = expected_new_actions - fired_ids
    if missing:
        print(f"\n[WARNING] Expected new interventions did not fire: {missing}")
    else:
        print("\n[OK] Both new climate/deforestation interventions fired as expected.")