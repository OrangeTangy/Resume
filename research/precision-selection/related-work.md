# Related-work audit (2026-09-19)
This is a targeted search, not an exhaustive systematic review. No priority or first-ever claim is justified.

1. Cobbe et al. Training Verifiers to Solve Math Word Problems (2021), https://arxiv.org/abs/2110.14168 . Establishes verifier selection over sampled math answers. Our pilot does not train a verifier or sample answer traces.
2. Gao, Schulman, Hilton. Scaling Laws for Reward Model Overoptimization (2022), https://arxiv.org/abs/2210.10760 . Establishes that optimizing an imperfect reward proxy can hurt the underlying objective, including best-of-N. Thus more candidates hurting correctness is not itself novel.
3. Zhang et al. Generative Verifiers: Reward Modeling as Next-Token Prediction (ICLR 2025), https://arxiv.org/abs/2408.15240 . Trains generative verifiers; the pilot uses a general instruction model prompted as a judge, not their trained method.
4. Zhou et al. Variation in Verification: Understanding Verification Dynamics in Large Language Models (ICLR 2026), https://arxiv.org/abs/2509.17995 . Examines task difficulty and generator/verifier capability. Important context for the weak-judge limitation of our 0.5B pilot.
5. Huang and Wen. Quasar: Quantized Self-Speculative Acceleration for Rapid Inference via Memory-Efficient Verification (2026 preprint), https://arxiv.org/abs/2603.01399 . Directly relevant low-precision verification, but token-level speculative decoding is different from scoring complete answers.
6. Qwen team. Qwen2.5-0.5B-Instruct model card, https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct . Source of the actual model used here.
7. MLX-LM, https://github.com/ml-explore/mlx-lm . Runtime used for actual weight-quantized inference on Apple silicon.

Candidate gap, not established novelty: joint effects of judge weight precision, candidate pool size, and selective rescoring on answer selection. A controlled one-model arithmetic pilot cannot establish generality or a PhD-level contribution.

Queries included combinations of quantization, reward model, verifier, best-of-N, ranking, precision, and reasoning selection. Abstracts, official model/runtime documentation, and available full HTML papers were inspected. Additional citation chasing, exact replication comparisons, and faculty/domain-expert review remain necessary before any conference submission.
