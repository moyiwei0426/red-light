# Dynamic Risk, Cognition, Attention, and Behavior Analysis

This project analyzes forced red-light crossing trials as a dynamic risk-response process:

`traffic risk -> cognitive load -> visual attention -> movement response`

It is intentionally separate from `ParticipantDatabase`. The database is read-only; all generated files are written under this project's `outputs` directory.

## Operational definitions

- Baseline: 3 seconds before `participant_entered_crosswalk`.
- Risk exposure: from `participant_entered_crosswalk` to `participant_entered_vehicle_lane`.
- Conflict: from `participant_entered_vehicle_lane` to `participant_reached_other_side`.
- Recovery: from `participant_reached_other_side` to `trial_end`.
- RPD: `(phase mean pupil - baseline mean pupil) / baseline mean pupil * 100`.
- Pupil cleaning: bilateral valid samples, 1.5-9.0 mm, inter-eye difference <= 1.5 mm, and rolling-median outlier removal.
- AOI dwell proportion: duration-weighted gaze samples in each AOI. This is not presented as a laboratory fixation classifier.
- Fixation-like bout: uninterrupted valid gaze in the same AOI for at least 100 ms.
- Gaze entropy: normalized Shannon entropy of AOI dwell proportions.
- Risk-response latency: time from the first dynamic TTC <= 3.5 s to a sustained speed change beyond the configured threshold.

The source data do not contain a dedicated `vehicle_passed` event. Recovery is therefore operationalized as the period after the participant reaches the other side until trial end. Every phase boundary is exported so the definition can be audited or changed.

## Run the pilot

```powershell
.\run_pilot.ps1
```

The default pilot uses F18, F19, and P16. To use a different Python executable:

```powershell
.\run_pilot.ps1 -PythonExe "C:\path\to\python.exe"
```

Required Python packages: `pandas`, `numpy`, and `Pillow`.

## Main outputs

- `tables/trial_metrics.csv`: one row per participant-condition trial; suitable for mixed models.
- `tables/phase_metrics.csv`: pupil, RPD, speed, TTC, AOI entropy, and gaze quality by phase.
- `tables/aoi_metrics.csv`: AOI dwell, fixation-like bouts, and proportions by phase.
- `tables/aoi_transitions.csv`: phase-specific AOI transitions.
- `tables/model_coefficients.csv`: pilot participant-fixed-effect regressions.
- `tables/quality_log.csv`: inclusion, synchronization, pupil, and phase quality checks.
- `figures/pilot_dashboard.png`: descriptive pilot visualization.
- `pilot_report.md`: short interpretation and limitations.

## Full-database run

Omit the participant filter to analyze every curated trial:

```powershell
& $PythonExe .\analyze.py --config .\config.json --output .\outputs\full_database --all-participants
```

The pilot regression is a pipeline check, not final inference. For the paper, use `trial_metrics.csv` or `phase_metrics.csv` in an LMM implementation with participant random intercepts and, where justified, random slopes.
