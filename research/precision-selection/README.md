# Weight Precision and Answer Selection

An AI-assisted, local-compute **research manuscript / exploratory pilot**. No claim of first-ever novelty or Google Scholar indexing.

## What was actually measured

Qwen2.5-0.5B-Instruct (exact revision in `model.json`) scores the same 1,920 constructed arithmetic-answer candidates at native BF16, 8-bit, and 4-bit weight precision. MLX performs real groupwise affine weight quantization, including quantizable embeddings and linear layers. There are 40 development and 80 held-out questions. The 16 answers per question are constructed, not LLM-generated; one is correct. Nested prefixes give N=1,2,4,8,16. Correctness comes from exact integer arithmetic.

`protocol.md` was written before scoring. The numerical batching control was added afterward and is explicitly labeled as a robustness check. All primary results, including null and adverse findings, are retained. This small general instruction model is not a trained reward model.

## Files

- `paper.pdf` and `paper.html`: manuscript with results, limitations, references, AI-assistance disclosure.
- `protocol.md`: experimental plan and interpretation boundaries.
- `data.json`: all prompts' questions, candidate answers, exact labels, and split assignments.
- `scores-*.json`: raw model scores, hashes, model revision, timing, and parameter storage.
- `analysis.json`, `question-results.csv`: complete summary and question-level results.
- `batching-control.json`: post-protocol native singleton-batch sensitivity control.
- `run_experiment.py`: dataset construction and MLX scoring.
- `analyze.py`: question-level paired bootstrap, all candidate counts, policy sweeps, and plots.
- `check_batching.py`: numerical sensitivity control.
- `build_paper.py`: PDF/HTML generator.
- `related-work.md`: targeted literature audit and novelty limits.
- `requirements-lock.txt`: exact environment packages used.

## Reproduction on Apple silicon

Use Python 3.11 in a fresh environment. Install the pinned dependencies with `python -m pip install -r STUDY/requirements-lock.txt`, replacing STUDY with this directory. Download the model revision specified in `model.json` into `work/model` with Hugging Face `snapshot_download`. The model weights are not redistributed in this artifact.

Run from the parent project folder, replacing `STUDY` below with this directory:

```sh
python STUDY/run_experiment.py --model work/model
python STUDY/run_experiment.py --model work/model --bits 8
python STUDY/run_experiment.py --model work/model --bits 4
python STUDY/analyze.py
python STUDY/check_batching.py
python STUDY/build_paper.py
```

To reproduce only statistics and figures from the included raw data, `analyze.py` needs NumPy and Matplotlib; model downloads and inference are unnecessary. `build_paper.py` needs ReportLab and the included results. Timing depends on hardware, software versions, batching and other laptop activity.

## Interpretation

The study isolates ranking behavior over fixed artificial candidate pools. It does not measure generated reasoning quality, deployed hybrid latency, energy, a production reward model, or performance on standardized reasoning benchmarks. Reference-selection agreement is not equivalent to correctness. Bootstrap intervals describe question-level variation within this constructed sample, not model-family or benchmark uncertainty.

## Authorship and status

Prepared for Tanay Anand using Codex for study design assistance, implementation, experiment execution, analysis, writing and website preparation.

## Exploratory 1.5B extension

After seeing the weak 0.5B judging baseline, the exact same experiment was extended to Qwen2.5-1.5B-Instruct. `replication-1.5b/` contains the unchanged candidate pools, an explicit protocol amendment, model revision, scorer, analysis and complete results. This is a same-family model-size extension on reused questions, not independent dataset replication. Run its scripts in the same order, pointing `--model` to the pinned 1.5B checkpoint. `validate.py` independently checks labels, selection counts and provenance hashes for both models.

## LaTeX manuscript

The PDF is typeset from paper.tex using Times New Roman with matching TeX Gyre Termes mathematics, the standard article class, numbered equations,
booktabs tables, and a BibTeX bibliography (references.bib).

With Tectonic installed, compile with: tectonic paper.tex
Alternatively, run XeLaTeX, BibTeX, then XeLaTeX twice. Install Times New Roman for an exact match; the source falls back to TeX Gyre Termes if it is unavailable. Fonts are not redistributed. The source archive
includes the three referenced figures with their relative directory paths.
To regenerate from measured JSON files, run: python build_paper.py
Set the TECTONIC environment variable to specify the executable if needed.
The published PDF was compiled with Tectonic 0.17.0.

Review: the author reports private review by colleagues; no journal or conference acceptance is specified.
