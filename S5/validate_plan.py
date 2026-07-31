#!/usr/bin/env python3
"""Validate the arithmetic and hard constraints in the S5 mixture plan."""

from __future__ import annotations

import json
from pathlib import Path


PLAN_PATH = Path(__file__).with_name("mixture-plan.json")


def format_indian(value: int) -> str:
    digits = str(value)
    if len(digits) <= 3:
        return digits
    head, tail = digits[:-3], digits[-3:]
    groups: list[str] = []
    while head:
        groups.append(head[-2:])
        head = head[:-2]
    return ",".join(reversed(groups)) + "," + tail


def require_sum(rows: list[dict], key: str, expected: float) -> None:
    actual = sum(row[key] for row in rows)
    if abs(actual - expected) > 1e-9:
        raise ValueError(f"{key} sums to {actual}, expected {expected}")


def main() -> int:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    lanes = plan["lanes"]
    indic = plan["indic_tiers"]
    agentic_subslots = plan["agentic_subslots"]
    agentic_provenance = plan["agentic_provenance"]

    require_sum(lanes, "target_percent", 100)
    require_sum(lanes, "anneal_percent", 100)
    require_sum(lanes, "protected_floor_percent", 61)
    require_sum(indic, "share_of_indic_percent", 100)
    require_sum(indic, "share_of_main_phase_percent", 12)
    require_sum(agentic_subslots, "share_of_agentic_percent", 100)
    require_sum(agentic_subslots, "share_of_main_phase_percent", 12)
    require_sum(agentic_provenance, "share_of_agentic_percent", 100)
    require_sum(plan["difficulty_bands"], "percent", 100)
    require_sum(plan["reasoning_length_bands"], "percent", 100)
    if sum(plan["anneal"]["indic_tiers_percent"].values()) != 100:
        raise ValueError("anneal Indic tiers do not sum to 100")
    if sum(plan["anneal"]["editable_guardrails_percent"].values()) != 17:
        raise ValueError("anneal editable guardrails do not sum to 17")

    for lane in lanes:
        if lane["protected_floor_percent"] > lane["target_percent"]:
            raise ValueError(f"floor exceeds target for {lane['id']}")

    budget = plan["model"]["training_token_budget"]
    main_budget = plan["model"]["main_pretraining_tokens"]
    reserve = plan["anneal"]["reserve_tokens"]
    expected_reserve = budget * plan["anneal"]["reserve_percent"] / 100
    if reserve != expected_reserve:
        raise ValueError("anneal reserve tokens do not match its percentage")
    if main_budget + reserve != budget:
        raise ValueError("main phase plus anneal reserve does not match total budget")

    realised = sum(
        main_budget * lane["target_percent"] / 100
        + reserve * lane["anneal_percent"] / 100
        for lane in lanes
    )
    if realised != budget:
        raise ValueError("realised lane exposures do not match total budget")

    print("S5 plan valid")
    print(f"lane target: {sum(row['target_percent'] for row in lanes):g}%")
    print(f"protected floor: {sum(row['protected_floor_percent'] for row in lanes):g}%")
    print(f"main pretraining: {format_indian(main_budget)} tokens")
    print(f"anneal reserve: {format_indian(reserve)} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
