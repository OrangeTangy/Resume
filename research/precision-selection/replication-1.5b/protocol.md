# Precision and selection: pilot protocol v1
Frozen before model scoring, 2026-09-19.

Status: exploratory local-compute pilot, not a preregistered study or peer-reviewed publication.
Question: Does weight quantization change best-of-N arithmetic-answer selection, and can margin-triggered full-precision rescoring recover the reference selection?

Model: Qwen/Qwen2.5-0.5B-Instruct, exact repository revision recorded on download. Compare native checkpoint precision, MLX affine groupwise 8-bit and 4-bit weight quantization, group size 64. These are distinct from output-score rounding. No fine-tuning.

Data: 120 procedurally constructed arithmetic questions, four templates, random seed 20260919. 40 development questions and 80 held-out questions; no question occurs in both. Each has 16 distinct integer answer candidates: one correct and 15 incorrect, randomly ordered once; use nested prefixes N=1,2,4,8,16. The answers are constructed, NOT sampled from an LLM. Oracle coverage therefore increases with N by construction and equals one at N=16. This is a controlled selection stress test, not an end-to-end reasoning benchmark. Candidate wording is identical except for the number. Score each candidate independently with prompted next-token logit(Yes)-logit(No). Validate labels are single tokens. Ground truth uses exact integer arithmetic, not reference-model judgments.

Primary analysis: per-question selected-answer correctness; paired quantized-minus-reference accuracy; top-selection disagreement; oracle answer availability. Use 10,000 paired question-level bootstrap resamples for 95% percentile intervals. Report every N and precision; no choosing a favorable subset. Also report harmful and beneficial selection flips separately. Do not treat nested N values or candidates as independent samples.

Rescoring: for each N>1, calibrate a top-two low-precision score-gap threshold on development questions only. Evaluate thresholds equal to development gap quantiles 0,.25,.5,.75,1, plus always/never rescoring. When the gap is below threshold, rescore ALL N candidates at reference precision. Report the whole tradeoff on held-out questions. Primary fixed policy uses the development median gap. Never use test correctness labels for threshold selection. A random-rescoring expected baseline uses the same held-out escalation fraction, calculated without choosing favorable random subsets.

Cost: record observed per-precision scoring wall times, model loading/quantization time, input tokens, model parameter storage. Cached-policy rescoring counts and estimated serial scoring cost are NOT measured deployment latency. No energy or speedup claims from bit widths. Peak process memory can be cumulative and is not per-model memory. Native and quantized passes load separately, on a shared laptop, so timings are descriptive.

Controls: same question/candidate pools and prompts at all precisions; deterministic model inference in eval mode; no KV quantization; batch size 4, right padding ignored by gathering each row's final real token. Warmup excluded from inference timing. Compare padded versus standalone scores on initial candidates to validate batching.

Interpretation: one tiny general instruction model may be a weak judge. No claim about trained reward models, frontier LLMs, chain-of-thought reasoning, or universal degradation. Any null or opposite result must remain in the report. The simple deterministic score-margin stability bound is a standard observation, not a novelty claim. Real-model multi-family replication and an exhaustive related-work review are needed for a stronger research contribution.

## Exploratory extension, defined after the 0.5B results and before 1.5B scoring
Use Qwen/Qwen2.5-1.5B-Instruct with all other data, scoring, quantization, and analysis settings unchanged. This same-family size extension addresses weak baseline judging. It is not an independent dataset replication, a preregistered confirmation, or a new test split. Preserve and report both models regardless of result direction. The native BF16 and 8-bit/4-bit runs must all complete before comparing conditions. Stop if local memory prevents safe execution; document failures rather than silently substitute models.

Execution amendment: the first native 1.5B attempt stalled under memory pressure after passing 640 candidates and was terminated without a completed score file. All three extension conditions are restarted with an MLX allocator cache limit of 128 MiB. This limit affects retained temporary buffers, not weights or quantization. The aborted log is retained. No partial outcomes were used to change the experimental question or candidate pools.
