"""Bridge connected_thinking's decision-making concept graph with the
DecisionMaking repo's quantitative bioventure success predictor.

Goal: help a human answer "is this a good research decision to go forward
with, or not?" by combining two different kinds of judgment:

  1. QUANTITATIVE: DecisionMaking's SuccessPredictor gives P(success), a
     GO/NO-GO verdict, safety profile, and historical failure lessons for a
     specific drug/program description.
  2. QUALITATIVE / PROCESS: connected_thinking's concept graph diffuses those
     quantitative signals through a decision-making subgraph (clear criteria,
     information gathering, bias awareness, reversibility, pre-mortems, ...)
     and produces a ranked checklist plus warnings about the process itself,
     not just the number.

The final recommendation is gated by both: a high P(success) alone is not
enough to say GO if the process activation shows the decision is being made
on thin evidence, bias, or sunk-cost pressure.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

DECISION_MAKING_ROOT = Path("/home2/DecisionMaking")
if str(DECISION_MAKING_ROOT) not in sys.path:
    sys.path.insert(0, str(DECISION_MAKING_ROOT))

from src.learning.decision_model import SuccessPredictor  # noqa: E402

from concept_graph import (  # noqa: E402
    build_yellow_lemon_graph,
    attach_decision_making_graph,
    decision_recommendations,
    generate_decision_theories,
)


def _quant_signals_to_seeds(explanation: dict[str, Any]) -> dict[str, float]:
    """Map the quantitative predictor's explain() output onto decision-graph seeds."""
    seeds: dict[str, float] = {}

    p = explanation["p_success"]
    seeds["expected_value"] = round((p - 0.5) * 2, 3)

    calibration_factors = {c["factor"] for c in explanation.get("calibration", [])}
    if "Validated molecular target" in calibration_factors:
        seeds["clear_criteria"] = seeds.get("clear_criteria", 0.0) + 0.5
    if "Unvalidated molecular target" in calibration_factors:
        seeds["risk_tolerance"] = seeds.get("risk_tolerance", 0.0) - 0.3
        seeds["pre_mortem"] = seeds.get("pre_mortem", 0.0) + 0.4
    if "Positive outcome signal in text" in calibration_factors:
        seeds["information_gathering"] = seeds.get("information_gathering", 0.0) + 0.4
    if "Failure / termination signal in text" in calibration_factors:
        seeds["bias_awareness"] = seeds.get("bias_awareness", 0.0) + 0.5
        seeds["sunk_cost"] = seeds.get("sunk_cost", 0.0) - 0.3
    if "Clear-cut tech-indication fit" in calibration_factors:
        seeds["clear_criteria"] = seeds.get("clear_criteria", 0.0) + 0.4
    if "Bleeding-edge tech in mismatched indication" in calibration_factors:
        seeds["second_order_effects"] = seeds.get("second_order_effects", 0.0) + 0.4
        seeds["risk_tolerance"] = seeds.get("risk_tolerance", 0.0) - 0.2

    lessons = explanation.get("historical_lessons") or []
    if lessons:
        seeds["pre_mortem"] = seeds.get("pre_mortem", 0.0) + 0.5
        seeds["diverse_perspectives"] = seeds.get("diverse_perspectives", 0.0) + 0.3
        if any(str(lesson.get("severity", "")).startswith("HIGH") for lesson in lessons):
            seeds["risk_tolerance"] = seeds.get("risk_tolerance", 0.0) - 0.4

    tier = str(explanation.get("safety_profile", {}).get("overall_risk_tier", "")).lower()
    if "high" in tier:
        seeds["risk_tolerance"] = seeds.get("risk_tolerance", 0.0) - 0.4
        seeds["reversibility"] = seeds.get("reversibility", 0.0) - 0.3  # treat as a one-way door

    seeds["decision_journal"] = seeds.get("decision_journal", 0.0) + 0.4  # always log the reasoning
    return seeds


def evaluate_research_program(row: dict[str, Any], db_path: Path | None = None) -> dict[str, Any]:
    """Train/query the quantitative predictor and fuse it with the qualitative
    decision-process graph. Returns a combined recommendation report."""
    predictor = SuccessPredictor()
    train_report = predictor.train(db_path=db_path)
    explanation = predictor.explain(row)

    graph = build_yellow_lemon_graph()
    decision_nodes = attach_decision_making_graph(graph)

    seeds = _quant_signals_to_seeds(explanation)
    activation = graph.diffuse(
        seeds, steps=8, decay=0.22, tunnel_threshold=0.008, target_rms=0.42, soft_cap=0.92
    )

    checklist = decision_recommendations(activation, top_k=8)
    theories = generate_decision_theories(activation)

    good_decision_score = activation.get("good_decision", 0.0)
    process_quality_score = activation.get("process_quality", 0.0)

    # Quantitative P(success) gates GO/NO-GO; process quality decides whether
    # a borderline signal is trustworthy enough to act on yet.
    p = explanation["p_success"]
    if p < 0.30:
        final = "NO-GO"
    elif p >= 0.55 and good_decision_score > 0.15:
        final = "GO"
    else:
        final = "CONTINUE -- gather more evidence before committing"

    return {
        "train_report": train_report,
        "quantitative": explanation,
        "decision_seeds": seeds,
        "process_activation": {k: activation[k] for k in decision_nodes},
        "process_quality": process_quality_score,
        "good_decision_score": good_decision_score,
        "checklist": checklist,
        "theories": theories,
        "final_recommendation": final,
    }


def print_report(report: dict[str, Any]) -> None:
    q = report["quantitative"]
    print("QUANTITATIVE MODEL (DecisionMaking / SuccessPredictor)")
    print(f"  P(success): {q['p_success']:.1%}   Verdict: {q['verdict']}")
    if report["train_report"].get("status") == "skipped_insufficient_data":
        print("  (using prior-based fallback -- not enough labelled data to fit ensemble)")
    for c in q.get("calibration", []):
        print(f"  - {c['factor']}: {c['adjustment']}")
    lessons = q.get("historical_lessons") or []
    if lessons:
        print("  Historical lessons:")
        for lesson in lessons[:3]:
            print(f"    - {lesson.get('transfer_note', lesson)}")

    print("\nQUALITATIVE PROCESS GRAPH (connected_thinking)")
    for name, score in sorted(report["process_activation"].items(), key=lambda x: -x[1])[:10]:
        print(f"  {name:20s} {score:+.3f}")

    print("\nDecision theories:")
    for i, t in enumerate(report["theories"], start=1):
        print(f"  {i}. {t}")

    print("\nActionable checklist:")
    for i, (name, score, tip) in enumerate(report["checklist"], start=1):
        print(f"  {i}. [{name}, {score:+.2f}] {tip}")

    print(f"\nFINAL RECOMMENDATION: {report['final_recommendation']}")


if __name__ == "__main__":
    example_row = {
        "indication": "oncology",
        "mechanism": "antibody",
        "clinical_stage": "phase1",
        "raw_text": (
            "HER2 Probody drug conjugate masked ADC CytomX protease-activated "
            "MMAE cleavable linker breast cancer phase 1 dose escalation"
        ),
    }
    db_path = DECISION_MAKING_ROOT / "data" / "bioventure.json"
    report = evaluate_research_program(example_row, db_path=db_path)
    print_report(report)
