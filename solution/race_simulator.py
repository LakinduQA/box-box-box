#!/usr/bin/env python3
"""
Phase 3 race simulator for Box Box Box.

Reads race input JSON from stdin and prints predicted finishing positions JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


COMPOUND_INDEX = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}


def load_params(params_path: Path | None = None) -> dict:
    if params_path is None:
        params_path = Path(__file__).resolve().parent / "params.json"

    with params_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_strategy_items(strategies: dict) -> List[Tuple[str, dict]]:
    return sorted(strategies.items(), key=lambda kv: int(kv[0].replace("pos", "")))


def build_pit_map(pit_stops: List[dict]) -> Dict[int, dict]:
    return {int(stop["lap"]): stop for stop in pit_stops}


def extract_advanced_features(race_config: dict, strategy: dict, pit_lap_uses_new_tire: bool, start_pos: int) -> List[float]:
    total_laps = int(race_config["total_laps"])
    base = float(race_config["base_lap_time"])
    temp = float(race_config["track_temp"])
    pit_lane_time = float(race_config["pit_lane_time"])

    pit_stops = strategy.get("pit_stops", [])
    pit_map = build_pit_map(pit_stops)

    laps = [0.0, 0.0, 0.0]
    ages = [0.0, 0.0, 0.0]
    stints = [0.0, 0.0, 0.0]

    current_tire = strategy["starting_tire"]
    tire_age = 0
    stints[COMPOUND_INDEX[current_tire]] += 1.0

    for lap in range(1, total_laps + 1):
        pit_stop = pit_map.get(lap)

        if pit_stop and pit_lap_uses_new_tire:
            current_tire = pit_stop["to_tire"]
            tire_age = 0
            stints[COMPOUND_INDEX[current_tire]] += 1.0

        idx = COMPOUND_INDEX[current_tire]
        laps[idx] += 1.0
        ages[idx] += float(tire_age)

        tire_age += 1

        if pit_stop and not pit_lap_uses_new_tire:
            current_tire = pit_stop["to_tire"]
            tire_age = 0
            stints[COMPOUND_INDEX[current_tire]] += 1.0

    pit_time = len(pit_stops) * pit_lane_time

    return [
        pit_time,
        float(start_pos),
        laps[0],
        laps[1],
        laps[2],
        ages[0],
        ages[1],
        ages[2],
        temp * laps[0],
        temp * laps[1],
        temp * laps[2],
        temp * ages[0],
        temp * ages[1],
        temp * ages[2],
        base * laps[0],
        base * laps[1],
        base * laps[2],
        base * ages[0],
        base * ages[1],
        base * ages[2],
        stints[0],
        stints[1],
        stints[2],
    ]


def simulate_with_advanced_model(race_payload: dict, params: dict) -> List[str]:
    race_config = race_payload["race_config"]
    strategies = race_payload["strategies"]

    model = params["advanced_model"]
    weights = model["weights"]
    pit_lap_uses_new_tire = bool(params.get("pit_lap_uses_new_tire", False))

    totals = []
    for pos_key, strategy in iter_strategy_items(strategies):
        start_pos = int(pos_key.replace("pos", ""))
        driver_id = strategy["driver_id"]
        features = extract_advanced_features(
            race_config=race_config,
            strategy=strategy,
            pit_lap_uses_new_tire=pit_lap_uses_new_tire,
            start_pos=start_pos,
        )

        score = 0.0
        for w, x in zip(weights, features):
            score += float(w) * float(x)

        totals.append((score, start_pos, driver_id))

    totals.sort(key=lambda row: (row[0], row[1]))
    return [row[2] for row in totals]


def extract_piecewise_age_features(race_config: dict, strategy: dict, max_age: int) -> List[float]:
    total_laps = int(race_config["total_laps"])
    pit_lane_time = float(race_config["pit_lane_time"])
    track_temp = float(race_config["track_temp"])

    pit_stops = strategy.get("pit_stops", [])
    pit_map = build_pit_map(pit_stops)

    hist = [[0.0 for _ in range(max_age + 1)] for _ in range(3)]
    laps = [0.0, 0.0, 0.0]

    current_tire = strategy["starting_tire"]
    tire_age = 0

    for lap in range(1, total_laps + 1):
        # Regulations semantics: age increments before lap calculation.
        tire_age += 1

        cidx = COMPOUND_INDEX[current_tire]
        age_bucket = tire_age if tire_age <= max_age else max_age
        hist[cidx][age_bucket] += 1.0
        laps[cidx] += 1.0

        # Pit happens at end of lap.
        pit_stop = pit_map.get(lap)
        if pit_stop is not None:
            current_tire = pit_stop["to_tire"]
            tire_age = 0

    flat_hist = []
    for cidx in range(3):
        flat_hist.extend(hist[cidx])

    temp_age_mass = [
        track_temp * sum(hist[0][1:]),
        track_temp * sum(hist[1][1:]),
        track_temp * sum(hist[2][1:]),
    ]

    pit_time = len(pit_stops) * pit_lane_time
    return [pit_time, laps[0], laps[2]] + flat_hist + temp_age_mass


def simulate_with_piecewise_age_model(race_payload: dict, params: dict) -> List[str]:
    race_config = race_payload["race_config"]
    strategies = race_payload["strategies"]

    model = params["advanced_model"]
    max_age = int(model["max_age"])
    weights = model["weights"]

    totals = []
    for pos_key, strategy in iter_strategy_items(strategies):
        start_pos = int(pos_key.replace("pos", ""))
        driver_id = strategy["driver_id"]
        features = extract_piecewise_age_features(race_config, strategy, max_age)

        score = 0.0
        for w, x in zip(weights, features):
            score += float(w) * float(x)

        totals.append((score, start_pos, driver_id))

    totals.sort(key=lambda row: (row[0], row[1]))
    return [row[2] for row in totals]


def simulate_driver_total_time(race_config: dict, strategy: dict, params: dict) -> float:
    total_laps = int(race_config["total_laps"])
    base_lap_time = float(race_config["base_lap_time"])
    pit_lane_time = float(race_config["pit_lane_time"])
    track_temp = float(race_config["track_temp"])

    compound_offset = params["compound_offset"]
    degradation_rate = params["degradation_rate"]
    temp_factor = float(params.get("temp_factor", 0.0))
    temp_degradation = params.get("temp_degradation", {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0})
    warmup_laps = params.get("degradation_warmup_laps", {"SOFT": 0, "MEDIUM": 0, "HARD": 0})
    pit_lap_uses_new_tire = bool(params.get("pit_lap_uses_new_tire", False))
    tire_age_first_lap = int(params.get("tire_age_first_lap", 0))

    pit_map = build_pit_map(strategy.get("pit_stops", []))
    current_tire = strategy["starting_tire"]
    tire_age = 0
    total_time = 0.0

    for lap in range(1, total_laps + 1):
        pit_stop = pit_map.get(lap)

        if pit_stop and pit_lap_uses_new_tire:
            current_tire = pit_stop["to_tire"]
            tire_age = 0

        # Regulations-driven age semantics: increment before lap-time calc when configured.
        if tire_age_first_lap == 1:
            tire_age += 1

        # Baseline lap pace for this track.
        lap_time = base_lap_time
        # Compound pace offset for current tire.
        lap_time += float(compound_offset[current_tire])
        # Linear degradation after compound warmup period.
        effective_age = max(0, tire_age - int(warmup_laps.get(current_tire, 0)))
        lap_time += float(degradation_rate[current_tire]) * effective_age
        # Temperature can further scale degradation by compound.
        lap_time += float(temp_degradation.get(current_tire, 0.0)) * track_temp * effective_age
        # Global temperature impact term.
        lap_time += temp_factor * track_temp

        if pit_stop:
            lap_time += pit_lane_time

        total_time += lap_time

        if tire_age_first_lap == 0:
            tire_age += 1

        if pit_stop and not pit_lap_uses_new_tire:
            current_tire = pit_stop["to_tire"]
            tire_age = 0

    return total_time


def predict_finishing_positions(race_payload: dict, params: dict) -> List[str]:
    if params.get("model_type") == "piecewise_age_pairwise_v1" and "advanced_model" in params:
        return simulate_with_piecewise_age_model(race_payload, params)

    if params.get("model_type") == "pairwise_linear_v2" and "advanced_model" in params:
        return simulate_with_advanced_model(race_payload, params)

    race_config = race_payload["race_config"]
    strategies = race_payload["strategies"]

    totals = []
    for pos_key, strategy in iter_strategy_items(strategies):
        start_pos = int(pos_key.replace("pos", ""))
        driver_id = strategy["driver_id"]
        total_time = simulate_driver_total_time(race_config, strategy, params)
        totals.append((total_time, start_pos, driver_id))

    # Deterministic tie-break: lower starting position wins equal total-time ties.
    totals.sort(key=lambda row: (row[0], row[1]))
    return [row[2] for row in totals]


def build_output(race_payload: dict, params: dict) -> dict:
    race_id = race_payload["race_id"]

    return {
        "race_id": race_id,
        "finishing_positions": predict_finishing_positions(race_payload, params),
    }


def main() -> None:
    race_payload = json.load(sys.stdin)
    params = load_params()
    output = build_output(race_payload, params)
    print(json.dumps(output))


if __name__ == "__main__":
    main()
