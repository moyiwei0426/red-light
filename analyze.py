import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PHASES = ["baseline", "risk_exposure", "conflict", "recovery"]
AOI_ORDER = [
    "DirectRisk",
    "RuleInformation",
    "IndirectOcclusionRisk",
    "SafetyFacility",
    "BackgroundEnvironment",
    "OtherAOI",
]
CONDITION_ORDER = ["C1", "C2", "C3", "C4", "C5", "C6"]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze dynamic pedestrian risk response.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--participants", nargs="*")
    parser.add_argument("--all-participants", action="store_true")
    return parser.parse_args()


def as_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def numeric(series):
    return pd.to_numeric(series, errors="coerce")


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def safe_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else math.nan


def safe_min(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.min()) if len(values) else math.nan


def safe_max(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.max()) if len(values) else math.nan


def normalize_aoi(value):
    text = "" if pd.isna(value) else str(value)
    compact = "".join(ch for ch in text.lower() if ch.isalnum())
    if not compact:
        return ""
    if "directrisk" in compact or "鐩存帴椋庨櫓" in text:
        return "DirectRisk"
    if "ruleinformation" in compact or "瑙勫垯淇℃伅" in text:
        return "RuleInformation"
    if (
        "indirectocclusion" in compact
        or "occlusionrisk" in compact
        or "闂存帴椋庨櫓" in text
        or "閬尅椋庨櫓" in text
    ):
        return "IndirectOcclusionRisk"
    if "safetyfacility" in compact or "瀹夊叏璁炬柦" in text:
        return "SafetyFacility"
    if (
        "backgroundenvironment" in compact
        or compact == "background"
        or "鑳屾櫙鐜" in text
    ):
        return "BackgroundEnvironment"
    return "OtherAOI"


def resolve_path(database, value):
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else database / path


def event_time(frame, event_name, fallback=math.nan):
    if "row_type" not in frame or "event_name" not in frame:
        return fallback
    rows = frame[(frame["row_type"] == "event") & (frame["event_name"] == event_name)]
    if rows.empty:
        return fallback
    value = pd.to_numeric(rows.iloc[0].get("elapsed_time"), errors="coerce")
    return float(value) if pd.notna(value) else fallback


def sample_durations(times):
    values = np.asarray(times, dtype=float)
    if not len(values):
        return np.array([], dtype=float)
    differences = np.diff(values)
    valid = differences[(differences > 0) & (differences <= 0.1)]
    default = float(np.median(valid)) if len(valid) else 1.0 / 90.0
    result = np.append(differences, default)
    return np.where((result > 0) & (result <= 0.1), result, default)


def duration_events(times, state, minimum_duration):
    times = np.asarray(times, dtype=float)
    state = np.asarray(state, dtype=bool)
    events = 0
    start = None
    for index, active in enumerate(state):
        if active and start is None:
            start = index
        if start is not None and (not active or index == len(state) - 1):
            end = index if not active else index + 1
            if end > start and times[end - 1] - times[start] >= minimum_duration:
                events += 1
            start = None
    return events


def sustained_response_time(times, changed, sustain_seconds):
    times = np.asarray(times, dtype=float)
    changed = np.asarray(changed, dtype=bool)
    start = None
    for index, active in enumerate(changed):
        if active and start is None:
            start = index
        if start is not None and (not active or index == len(changed) - 1):
            end = index if not active else index + 1
            if end > start and times[end - 1] - times[start] >= sustain_seconds:
                return float(times[start])
            start = None
    return math.nan


def build_bouts(times, categories, valid, minimum_duration):
    bouts = []
    active = ""
    start = math.nan
    last = math.nan
    for time_value, category, is_valid in zip(times, categories, valid):
        category = category if is_valid else ""
        gap = time_value - last if finite(last) else 0.0
        if category != active or gap > 0.1:
            if active and finite(start) and finite(last):
                duration = max(0.0, last - start)
                if duration >= minimum_duration:
                    bouts.append((active, float(start), float(last), duration))
            active = category
            start = float(time_value) if category else math.nan
        last = float(time_value)
    if active and finite(start) and finite(last):
        duration = max(0.0, last - start)
        if duration >= minimum_duration:
            bouts.append((active, float(start), float(last), duration))
    return bouts


def phase_name(times, bounds):
    result = np.full(len(times), "outside", dtype=object)
    for name in PHASES:
        start, end = bounds[name]
        if finite(start) and finite(end) and end > start:
            result[(times >= start) & (times < end)] = name
    return result


def clean_pupil(eye, config):
    left = numeric(eye.get("left_pupil_diameter_mm", pd.Series(index=eye.index, dtype=float)))
    right = numeric(eye.get("right_pupil_diameter_mm", pd.Series(index=eye.index, dtype=float)))
    left_valid = numeric(eye.get("left_pupil_diameter_valid", 0)).fillna(0).eq(1)
    right_valid = numeric(eye.get("right_pupil_diameter_valid", 0)).fillna(0).eq(1)
    minimum = float(config["pupil_min_mm"])
    maximum = float(config["pupil_max_mm"])
    valid = (
        left_valid
        & right_valid
        & left.between(minimum, maximum)
        & right.between(minimum, maximum)
        & (left - right).abs().le(float(config["pupil_max_inter_eye_difference_mm"]))
    )
    pupil = ((left + right) / 2.0).where(valid)
    rolling = pupil.rolling(9, center=True, min_periods=3).median()
    pupil = pupil.where((pupil - rolling).abs().le(float(config["pupil_rolling_outlier_mm"])) | rolling.isna())
    return pupil


def read_behavior(path, indexed_row, config):
    frame = pd.read_csv(path, low_memory=False)
    for column in [
        "elapsed_time",
        "unity_frame",
        "participant_speed",
        "nearest_vehicle_time_to_arrival_reference",
        "nearest_vehicle_distance_to_participant",
        "nearest_vehicle_speed",
        "distance_to_crosswalk_end",
    ]:
        if column in frame:
            frame[column] = numeric(frame[column])

    trial_end_fallback = numeric(frame.get("elapsed_time", pd.Series(dtype=float))).max()
    crosswalk = event_time(frame, "participant_entered_crosswalk", indexed_row.get("crosswalk_entry_time_s"))
    road = event_time(frame, "participant_entered_vehicle_lane", indexed_row.get("road_entry_time_s"))
    other = event_time(frame, "participant_reached_other_side", indexed_row.get("other_side_time_s"))
    trial_end = event_time(frame, "trial_end", trial_end_fallback)
    baseline_start = crosswalk - float(config["baseline_seconds"]) if finite(crosswalk) else math.nan
    bounds = {
        "baseline": (max(0.0, baseline_start), crosswalk),
        "risk_exposure": (crosswalk, road),
        "conflict": (road, other),
        "recovery": (other, trial_end + 1e-6),
    }

    samples = frame[frame.get("row_type", "") == "sample"].copy()
    samples = samples.dropna(subset=["elapsed_time"]).sort_values("elapsed_time")
    times = samples["elapsed_time"].to_numpy(dtype=float)
    speeds = numeric(samples.get("participant_speed", np.nan)).to_numpy(dtype=float)
    samples["phase"] = phase_name(times, bounds)
    smooth_speed = pd.Series(speeds).rolling(5, center=True, min_periods=2).median().to_numpy()
    acceleration = np.full(len(times), np.nan)
    valid_motion = np.isfinite(times) & np.isfinite(smooth_speed)
    if valid_motion.sum() >= 3:
        acceleration[valid_motion] = np.gradient(smooth_speed[valid_motion], times[valid_motion])
    samples["participant_acceleration"] = acceleration

    def speed_in(name):
        return safe_mean(samples.loc[samples["phase"] == name, "participant_speed"])

    baseline_speed = speed_in("baseline")
    conflict_speed = speed_in("conflict")
    ttc = numeric(samples.get("nearest_vehicle_time_to_arrival_reference", np.nan)).to_numpy(dtype=float)
    risk_candidates = np.flatnonzero(
        np.isfinite(ttc)
        & np.isfinite(times)
        & (ttc <= float(config["ttc_risk_threshold_seconds"]))
        & (times >= max(0.0, crosswalk if finite(crosswalk) else 0.0))
    )
    risk_onset = float(times[risk_candidates[0]]) if len(risk_candidates) else math.nan
    response_time = math.nan
    if finite(risk_onset):
        pre = (times >= risk_onset - 1.0) & (times < risk_onset) & np.isfinite(speeds)
        reference_speed = safe_mean(speeds[pre])
        threshold = max(
            float(config["speed_response_absolute_threshold_mps"]),
            abs(reference_speed) * float(config["speed_response_relative_threshold"])
            if finite(reference_speed)
            else 0.0,
        )
        post = times >= risk_onset
        changed = post & np.isfinite(speeds)
        if finite(reference_speed):
            changed &= np.abs(speeds - reference_speed) >= threshold
        response_time = sustained_response_time(
            times, changed, float(config["speed_response_sustain_seconds"])
        )

    crossing_window = (
        (times >= crosswalk) & (times <= other)
        if finite(crosswalk) and finite(other)
        else np.zeros(len(times), dtype=bool)
    )
    stopped = crossing_window & np.isfinite(speeds) & (speeds < float(config["stop_speed_threshold_mps"]))
    distance_end = numeric(samples.get("distance_to_crosswalk_end", np.nan)).to_numpy(dtype=float)
    retreat = np.zeros(len(times), dtype=bool)
    valid_distance = np.isfinite(distance_end) & np.isfinite(times)
    if valid_distance.sum() >= 3:
        smooth_distance = (
            pd.Series(distance_end[valid_distance])
            .rolling(5, center=True, min_periods=2)
            .median()
            .to_numpy(dtype=float)
        )
        derivative = np.gradient(smooth_distance, times[valid_distance])
        retreat_indices = np.flatnonzero(valid_distance)
        retreat[retreat_indices] = derivative > 0.15
        retreat &= crossing_window

    entry_rows = frame[(frame.get("row_type", "") == "event") & (frame.get("event_name", "") == "participant_entered_vehicle_lane")]
    entry_ttc = (
        pd.to_numeric(entry_rows.iloc[0].get("nearest_vehicle_time_to_arrival_reference"), errors="coerce")
        if not entry_rows.empty
        else math.nan
    )
    metrics = {
        "crosswalk_entry_s": crosswalk,
        "road_entry_s": road,
        "other_side_s": other,
        "trial_end_s": trial_end,
        "baseline_start_s": bounds["baseline"][0],
        "actual_entry_ttc_s": float(entry_ttc) if pd.notna(entry_ttc) else math.nan,
        "minimum_dynamic_ttc_s": safe_min(ttc[ttc >= 0]),
        "baseline_speed_mps": baseline_speed,
        "risk_speed_mps": speed_in("risk_exposure"),
        "conflict_speed_mps": conflict_speed,
        "recovery_speed_mps": speed_in("recovery"),
        "delta_speed_conflict_vs_baseline_mps": conflict_speed - baseline_speed
        if finite(conflict_speed) and finite(baseline_speed)
        else math.nan,
        "maximum_deceleration_mps2": safe_min(acceleration[(times >= road) & (times <= other)])
        if finite(road) and finite(other)
        else math.nan,
        "risk_onset_s": risk_onset,
        "speed_response_time_s": response_time,
        "response_latency_s": response_time - risk_onset
        if finite(response_time) and finite(risk_onset)
        else math.nan,
        "stop_count": duration_events(times, stopped, float(config["stop_min_duration_seconds"])),
        "retreat_count": duration_events(times, retreat, float(config["stop_min_duration_seconds"])),
    }
    frame_map = samples[["unity_frame", "elapsed_time", "phase"]].dropna(subset=["unity_frame"]).copy()
    frame_map["unity_frame"] = frame_map["unity_frame"].astype(int)
    frame_map = frame_map.sort_values("unity_frame").drop_duplicates("unity_frame")
    return metrics, samples, bounds, frame_map


def read_eye(path, frame_map, bounds, config):
    columns = [
        "unity_frame",
        "left_pupil_diameter_valid",
        "right_pupil_diameter_valid",
        "left_pupil_diameter_mm",
        "right_pupil_diameter_mm",
        "focus_hit",
        "focus_aoi_category",
    ]
    eye = pd.read_csv(path, usecols=lambda column: column in columns, low_memory=False)
    eye["unity_frame"] = numeric(eye["unity_frame"])
    eye = eye.dropna(subset=["unity_frame"]).sort_values("unity_frame")
    eye["unity_frame"] = eye["unity_frame"].astype(int)
    sync = pd.merge_asof(
        eye,
        frame_map[["unity_frame", "elapsed_time"]].sort_values("unity_frame"),
        on="unity_frame",
        direction="nearest",
        tolerance=2,
    )
    times = numeric(sync["elapsed_time"]).to_numpy(dtype=float)
    sync["phase"] = phase_name(times, bounds)
    sync["pupil_mm"] = clean_pupil(sync, config)
    sync["aoi"] = sync.get("focus_aoi_category", "").map(normalize_aoi)
    focus_valid = numeric(sync.get("focus_hit", 0)).fillna(0).eq(1)
    sync.loc[~focus_valid, "aoi"] = ""
    sync["sample_duration_s"] = sample_durations(times)
    return sync


def eye_metrics(sync, participant, condition, baseline_mean, config):
    phase_rows = []
    aoi_rows = []
    transition_rows = []
    valid_sync = sync["elapsed_time"].notna()
    total_rows = len(sync)
    for phase in PHASES:
        group = sync[(sync["phase"] == phase) & valid_sync].copy()
        pupil = numeric(group["pupil_mm"])
        valid_aoi = group["aoi"].ne("")
        aoi_duration = group.loc[valid_aoi].groupby("aoi")["sample_duration_s"].sum()
        total_aoi_duration = float(aoi_duration.sum())
        proportions = aoi_duration / total_aoi_duration if total_aoi_duration > 0 else aoi_duration
        entropy = math.nan
        positive = proportions[proportions > 0].to_numpy(dtype=float)
        if len(positive) > 1:
            entropy = float(-(positive * np.log(positive)).sum() / math.log(len(positive)))
        phase_mean = safe_mean(pupil)
        rpd = (
            (phase_mean - baseline_mean) / baseline_mean * 100.0
            if finite(phase_mean) and finite(baseline_mean) and baseline_mean > 0
            else math.nan
        )
        bouts = build_bouts(
            numeric(group["elapsed_time"]).to_numpy(dtype=float),
            group["aoi"].to_numpy(dtype=object),
            valid_aoi.to_numpy(dtype=bool),
            float(config["fixation_min_duration_seconds"]),
        )
        phase_rows.append(
            {
                "participant_id": participant,
                "condition_id": condition,
                "phase": phase,
                "phase_rows": len(group),
                "pupil_valid_rows": int(pupil.notna().sum()),
                "pupil_valid_ratio": float(pupil.notna().mean()) if len(group) else math.nan,
                "mean_pupil_mm": phase_mean,
                "baseline_mean_pupil_mm": baseline_mean,
                "rpd_pct": rpd,
                "aoi_valid_duration_s": total_aoi_duration,
                "normalized_gaze_entropy": entropy,
                "fixation_like_bout_count": len(bouts),
            }
        )
        for aoi in AOI_ORDER:
            selected_bouts = [bout for bout in bouts if bout[0] == aoi]
            aoi_rows.append(
                {
                    "participant_id": participant,
                    "condition_id": condition,
                    "phase": phase,
                    "aoi": aoi,
                    "dwell_duration_s": float(aoi_duration.get(aoi, 0.0)),
                    "dwell_proportion": float(proportions.get(aoi, 0.0)) if total_aoi_duration else math.nan,
                    "fixation_like_bout_count": len(selected_bouts),
                    "mean_fixation_like_bout_s": safe_mean([bout[3] for bout in selected_bouts]),
                }
            )
        bout_categories = [bout[0] for bout in bouts]
        for source, target in zip(bout_categories, bout_categories[1:]):
            if source != target:
                transition_rows.append(
                    {
                        "participant_id": participant,
                        "condition_id": condition,
                        "phase": phase,
                        "source_aoi": source,
                        "target_aoi": target,
                    }
                )
    return phase_rows, aoi_rows, transition_rows, total_rows, int(valid_sync.sum())


def fit_participant_fixed_effect(frame, model_name, outcome, predictors):
    columns = ["participant_id", outcome] + predictors
    data = frame[columns].copy()
    for column in [outcome] + predictors:
        data[column] = numeric(data[column])
    data = data.dropna()
    if len(data) < len(predictors) + 3 or data["participant_id"].nunique() < 2:
        return [{"model": model_name, "term": "not_fitted", "n": len(data), "reason": "insufficient_data"}]
    participant_dummies = pd.get_dummies(data["participant_id"], prefix="participant", drop_first=True, dtype=float)
    x_frame = pd.concat([pd.Series(1.0, index=data.index, name="Intercept"), data[predictors], participant_dummies], axis=1)
    x = x_frame.to_numpy(dtype=float)
    y = data[outcome].to_numpy(dtype=float)
    coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ coefficients
    residual = y - fitted
    degrees = max(1, len(y) - rank)
    variance = float(residual @ residual / degrees)
    covariance = variance * np.linalg.pinv(x.T @ x)
    standard_errors = np.sqrt(np.maximum(0.0, np.diag(covariance)))
    total_sum = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - float((residual**2).sum()) / total_sum if total_sum > 0 else math.nan
    rows = []
    for term, coefficient, standard_error in zip(x_frame.columns, coefficients, standard_errors):
        rows.append(
            {
                "model": model_name,
                "outcome": outcome,
                "term": term,
                "coefficient": float(coefficient),
                "standard_error": float(standard_error),
                "n": len(data),
                "participant_count": data["participant_id"].nunique(),
                "r_squared": r_squared,
                "note": "pilot participant fixed effects; not final LMM inference",
            }
        )
    return rows


def font(size=24):
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def draw_panel(draw, box, title, points, x_labels, y_label, color):
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=14, fill="#FFFFFF", outline="#D0D5DD", width=2)
    draw.text((left + 20, top + 14), title, fill="#17324D", font=font(25))
    plot = (left + 70, top + 65, right - 25, bottom - 70)
    x0, y0, x1, y1 = plot
    draw.line((x0, y0, x0, y1), fill="#667085", width=2)
    draw.line((x0, y1, x1, y1), fill="#667085", width=2)
    finite_points = [(index, value) for index, value in enumerate(points) if finite(value)]
    if not finite_points:
        draw.text((x0 + 30, y0 + 60), "No valid values", fill="#667085", font=font(20))
        return
    values = [value for _, value in finite_points]
    minimum = min(values)
    maximum = max(values)
    padding = max(0.1, (maximum - minimum) * 0.2)
    minimum -= padding
    maximum += padding
    coordinates = []
    for index, value in finite_points:
        x = x0 + (x1 - x0) * index / max(1, len(points) - 1)
        y = y1 - (y1 - y0) * (value - minimum) / max(1e-9, maximum - minimum)
        coordinates.append((x, y))
    if len(coordinates) > 1:
        draw.line(coordinates, fill=color, width=4)
    for (x, y), (_, value) in zip(coordinates, finite_points):
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)
        draw.text((x - 22, y - 31), f"{value:.2f}", fill="#344054", font=font(15))
    for index, label in enumerate(x_labels):
        x = x0 + (x1 - x0) * index / max(1, len(x_labels) - 1)
        draw.text((x - 12, y1 + 12), label, fill="#344054", font=font(17))
    draw.text((left + 12, top + 58), y_label, fill="#667085", font=font(15))


def make_dashboard(trials, phases, aoi, output_path):
    image = Image.new("RGB", (1700, 1080), "#F4F7FA")
    draw = ImageDraw.Draw(image)
    draw.text((55, 30), "Dynamic Risk Response Pilot", fill="#17324D", font=font(38))
    draw.text(
        (57, 80),
        f"{trials['participant_id'].nunique()} participants | {len(trials)} trials | descriptive pipeline check",
        fill="#667085",
        font=font(20),
    )
    conflict = phases[phases["phase"] == "conflict"]
    direct = aoi[(aoi["phase"] == "conflict") & (aoi["aoi"] == "DirectRisk")]
    rpd = conflict.groupby("condition_id")["rpd_pct"].mean().reindex(CONDITION_ORDER)
    dwell = direct.groupby("condition_id")["dwell_proportion"].mean().reindex(CONDITION_ORDER)
    speed = trials.groupby("condition_id")["delta_speed_conflict_vs_baseline_mps"].mean().reindex(CONDITION_ORDER)
    draw_panel(draw, (45, 125, 825, 555), "Conflict-phase relative pupil dilation", rpd.tolist(), CONDITION_ORDER, "RPD (%)", "#D92D20")
    draw_panel(draw, (875, 125, 1655, 555), "Conflict attention to direct risk", dwell.tolist(), CONDITION_ORDER, "Proportion", "#1570EF")
    draw_panel(draw, (45, 600, 825, 1030), "Movement speed adjustment", speed.tolist(), CONDITION_ORDER, "Delta m/s", "#039855")
    quality = trials.groupby("condition_id")["eye_sync_ratio"].mean().reindex(CONDITION_ORDER)
    draw_panel(draw, (875, 600, 1655, 1030), "Eye-behavior synchronization coverage", quality.tolist(), CONDITION_ORDER, "Ratio", "#7A5AF8")
    image.save(output_path)


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    project_dir = config_path.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    database = Path(config["database"])
    if not database.is_absolute():
        database = (project_dir / database).resolve()
    output = Path(args.output).resolve()
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    index_path = database / "indexes" / "selected_latest_valid_by_condition.csv"
    indexed = pd.read_csv(index_path, low_memory=False)
    indexed = indexed[
        indexed["complete_crossing"].map(as_bool)
        & indexed["include_in_curated_database"].map(as_bool)
    ].copy()
    if not args.all_participants:
        participants = args.participants or config["pilot_participants"]
        indexed = indexed[indexed["participant_id"].isin(participants)].copy()
    indexed["condition_id"] = indexed["condition_id"].astype(str)
    indexed = indexed.sort_values(["participant_id", "condition_id"])

    trial_rows = []
    phase_rows = []
    aoi_rows = []
    transition_rows = []
    quality_rows = []
    for _, indexed_row in indexed.iterrows():
        participant = str(indexed_row["participant_id"])
        condition = str(indexed_row["condition_id"])
        behavior_path = resolve_path(database, indexed_row.get("behavior_backup_file"))
        eye_path = resolve_path(database, indexed_row.get("eye_backup_file"))
        base_quality = {
            "participant_id": participant,
            "condition_id": condition,
            "behavior_file": str(behavior_path or ""),
            "eye_file": str(eye_path or ""),
        }
        if behavior_path is None or eye_path is None or not behavior_path.exists() or not eye_path.exists():
            quality_rows.append({**base_quality, "status": "excluded", "reason": "missing_input_file"})
            continue
        try:
            behavior_metrics, behavior_samples, bounds, frame_map = read_behavior(behavior_path, indexed_row, config)
            sync = read_eye(eye_path, frame_map, bounds, config)
            baseline_pupil = safe_mean(sync.loc[sync["phase"] == "baseline", "pupil_mm"])
            phases, aois, transitions, eye_rows, synced_rows = eye_metrics(
                sync, participant, condition, baseline_pupil, config
            )
            for row in phases:
                phase_behavior = behavior_samples[behavior_samples["phase"] == row["phase"]]
                row["mean_speed_mps"] = safe_mean(phase_behavior.get("participant_speed", []))
                row["minimum_ttc_s"] = safe_min(
                    numeric(phase_behavior.get("nearest_vehicle_time_to_arrival_reference", pd.Series(dtype=float)))
                )
                phase_rows.append(row)
            aoi_rows.extend(aois)
            transition_rows.extend(transitions)
            conflict_phase = next((row for row in phases if row["phase"] == "conflict"), {})
            conflict_direct = next(
                (row for row in aois if row["phase"] == "conflict" and row["aoi"] == "DirectRisk"),
                {},
            )
            trial_rows.append(
                {
                    "participant_id": participant,
                    "condition_id": condition,
                    "condition_vehicle_speed_kmh": pd.to_numeric(indexed_row.get("condition_vehicle_speed_kmh"), errors="coerce"),
                    "condition_ttc_s": pd.to_numeric(indexed_row.get("condition_ttc_s"), errors="coerce"),
                    "condition_risk_code": indexed_row.get("condition_risk_code", ""),
                    **behavior_metrics,
                    "baseline_mean_pupil_mm": baseline_pupil,
                    "conflict_rpd_pct": conflict_phase.get("rpd_pct", math.nan),
                    "conflict_gaze_entropy": conflict_phase.get("normalized_gaze_entropy", math.nan),
                    "conflict_directrisk_proportion": conflict_direct.get("dwell_proportion", math.nan),
                    "eye_rows": eye_rows,
                    "eye_synced_rows": synced_rows,
                    "eye_sync_ratio": synced_rows / eye_rows if eye_rows else math.nan,
                    "head_not_tracked_rows": pd.to_numeric(indexed_row.get("tracking_head_not_tracked_rows"), errors="coerce"),
                    "source_quality_status": indexed_row.get("quality_status", ""),
                }
            )
            baseline_valid = finite(baseline_pupil)
            phase_complete = all(
                finite(bounds[name][0]) and finite(bounds[name][1]) and bounds[name][1] > bounds[name][0]
                for name in PHASES
            )
            quality_rows.append(
                {
                    **base_quality,
                    "status": "included" if baseline_valid else "included_with_warning",
                    "reason": "ok" if baseline_valid else "invalid_pupil_baseline",
                    "eye_sync_ratio": synced_rows / eye_rows if eye_rows else math.nan,
                    "baseline_pupil_valid": baseline_valid,
                    "all_phase_boundaries_valid": phase_complete,
                }
            )
        except Exception as error:
            quality_rows.append({**base_quality, "status": "excluded", "reason": f"processing_error:{error}"})

    trials = pd.DataFrame(trial_rows)
    phases = pd.DataFrame(phase_rows)
    aois = pd.DataFrame(aoi_rows)
    transitions = pd.DataFrame(transition_rows)
    quality = pd.DataFrame(quality_rows)
    if trials.empty:
        raise RuntimeError("No trials were processed. Review tables/quality_log.csv.")

    model_rows = []
    model_rows.extend(
        fit_participant_fixed_effect(
            trials,
            "RQ1_cognitive_load",
            "conflict_rpd_pct",
            ["actual_entry_ttc_s", "condition_vehicle_speed_kmh"],
        )
    )
    model_rows.extend(
        fit_participant_fixed_effect(
            trials,
            "RQ2_visual_attention",
            "conflict_directrisk_proportion",
            ["actual_entry_ttc_s", "condition_vehicle_speed_kmh"],
        )
    )
    model_rows.extend(
        fit_participant_fixed_effect(
            trials,
            "RQ3_behavior_adjustment",
            "delta_speed_conflict_vs_baseline_mps",
            ["conflict_rpd_pct", "actual_entry_ttc_s"],
        )
    )
    models = pd.DataFrame(model_rows)

    trials.to_csv(tables / "trial_metrics.csv", index=False, encoding="utf-8-sig")
    phases.to_csv(tables / "phase_metrics.csv", index=False, encoding="utf-8-sig")
    aois.to_csv(tables / "aoi_metrics.csv", index=False, encoding="utf-8-sig")
    transitions.to_csv(tables / "aoi_transitions.csv", index=False, encoding="utf-8-sig")
    models.to_csv(tables / "model_coefficients.csv", index=False, encoding="utf-8-sig")
    quality.to_csv(tables / "quality_log.csv", index=False, encoding="utf-8-sig")
    make_dashboard(trials, phases, aois, figures / "pilot_dashboard.png")

    included = int((quality["status"].astype(str).str.startswith("included")).sum())
    baseline_valid = int(quality.get("baseline_pupil_valid", pd.Series(dtype=bool)).fillna(False).sum())
    conflict_rpd = safe_mean(trials["conflict_rpd_pct"])
    direct_risk = safe_mean(trials["conflict_directrisk_proportion"])
    delta_speed = safe_mean(trials["delta_speed_conflict_vs_baseline_mps"])
    report = [
        "# Pilot analysis report",
        "",
        "## Scope",
        "",
        f"- Participants: {trials['participant_id'].nunique()}",
        f"- Processed trials: {len(trials)}",
        f"- Included quality records: {included}",
        f"- Trials with valid pupil baseline: {baseline_valid}/{len(trials)}",
        "",
        "## Descriptive pipeline check",
        "",
        f"- Mean conflict-phase RPD: {conflict_rpd:.2f}%" if finite(conflict_rpd) else "- Mean conflict-phase RPD: unavailable",
        f"- Mean conflict direct-risk AOI proportion: {direct_risk:.3f}" if finite(direct_risk) else "- Mean conflict direct-risk AOI proportion: unavailable",
        f"- Mean conflict-versus-baseline speed change: {delta_speed:.3f} m/s" if finite(delta_speed) else "- Mean conflict-versus-baseline speed change: unavailable",
        "",
        "## Interpretation limits",
        "",
        "This pilot verifies extraction, synchronization, phase segmentation, and model-ready output. With only three participants, coefficient direction and magnitude must not be treated as confirmatory evidence. Use the full curated database and mixed-effects models for paper-level inference.",
        "",
        "AOI dwell proportions and fixation-like bouts are derived from recorded AOI samples. They should not be described as a validated I-VT/I-DT fixation classifier unless a dedicated fixation algorithm is added.",
    ]
    (output / "pilot_report.md").write_text("\n".join(report), encoding="utf-8")
    summary = {
        "database": str(database),
        "output": str(output),
        "participants": sorted(trials["participant_id"].unique().tolist()),
        "processed_trials": len(trials),
        "valid_pupil_baseline_trials": baseline_valid,
        "database_was_read_only": True,
    }
    (output / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
