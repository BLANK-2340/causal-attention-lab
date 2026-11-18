# Causal Attention Lab

A small CPU language-model experiment with **manual forward and backward passes**, not an autodiff wrapper. The question is deliberately narrow: can a causal attention head copy an earlier subject colour despite a distractor colour?

```text
the red robot near the blue lamp stays red .
```

This is a finite synthetic grammar, not a competitive language model or natural-language benchmark. The implementation is original; the attention and optimizer ideas are attributed below.

## Run

Tested with Python 3.12.14 and NumPy 2.3.5. The pinned environment targets Python 3.12; the earlier core-only module's Python 3.10 claim is not a tested installation combination with this NumPy pin.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1
python -W error tiny_lm.py test
python -W error -m unittest -v test_workbench
python -W error workbench.py check results/study.json
```

## Live local generation demo

```bash
python workbench.py train local-checkpoint.json
python workbench.py generate local-checkpoint.json --prompt 'the blue rover'
python workbench.py inspect local-checkpoint.json --prompt 'the red robot near the blue lamp stays'
```

Observed generated output:

```text
<bos> the blue rover near the blue lamp stays blue . <eos>
```

`train` fits the seed-7 model for 600 updates. `generate` and `inspect` load its weights and perform real CPU inference without retraining. `inspect` reports next-word probabilities and attention weights for the final input position. Attention weights are descriptive, not causal explanations. There is no public browser demo for this project; the three earlier portfolio demos are separate projects.

Checkpoint destinations must not exist. The JSON loader checks size, duplicate keys, format, exact vocabulary, core-source hash, checksum, parameter names/shapes/types and finite bounded numbers. No pickle or executable deserialization is used. Checksums detect accidental corruption, not authenticity. Version 1 accepts only this bounded 16-dimensional model and its no-attention ablation. Adam state is not saved: these are inference checkpoints, not resumable training snapshots. A changed core source requires a new checkpoint.

## Architecture

`tiny_lm.py` contains the mathematical implementation and independent tests. `workbench.py` adds persistence, inference inspection and evidence regeneration; it does not replace the model.

For each position, token and learned absolute-position embeddings form `X`. Projections give `Q`, `K`, `V`; attention is `softmax(QKᵀ / sqrt(16) + causal_mask) V`. The mask forbids future positions but permits the current token. A residual adds `X`, then a linear vocabulary projection produces next-token logits. Stable log-softmax cross entropy is averaged across batch and time. Explicit gradients include the softmax Jacobian-vector product, both Q/K paths, residual path and repeated-token embedding accumulation.

There is one head and one block, with no feed-forward network, layer normalization, dropout or encoder. This is not a full Transformer reproduction. Adam uses rate .015, beta1 .9, beta2 .999, epsilon 1e-8, 600 updates and batches of 32 distinct training sentences per update. No test-based stopping or tuning is performed.

## Protocol and results

The original corpus enumerates 256 sentences: four subject colours × four subjects × four distractor colours × four landmarks. Seed 90210 partitions **complete sentences** into 192 training and 64 held-out combinations. The vocabulary and grammar are specified before fitting. Eleven shifted next-token targets include EOS and exclude BOS. The repeated-colour prediction is target index 8.

Seeds 7, 42 and 2026 change initialization and batch order, not the split. Each compares attention with a separately trained token+position-only ablation using paired initialization/batch streams. The attention model has 1538 active parameters; the ablation has 770 active parameters and 768 unused stored Q/K/V entries. This is not a capacity-matched comparison. Add-0.5 unigram/bigram references use only training counts.

| Method | Held-out nats/token, seeds 7 / 42 / 2026 | Correct repeated colour, out of 64 |
|---|---|---|
| Attention | .538263 / .539656 / .530634 | 64 / 64 / 64 |
| No attention | .661639 / .661858 / .642432 | 14 / 13 / 13 |
| Bigram | 1.004598, one deterministic fit | 13 |
| Unigram | 2.609949, one deterministic fit | 0 |

The unigram's globally most frequent predicted word is `the`, so its agreement accuracy is zero, not a colour-only chance estimate. Attention's full token accuracy is about 69.9–71.6%, not 100%: several next words are intrinsically ambiguous before generation. Full-precision losses, perplexity, train/held-out token accuracy, samples and active/stored parameter counts are in [results/study.json](results/study.json).

Recompute without overwriting any saved evidence:

```bash
python tiny_lm.py experiment
python workbench.py export fresh-study.json
python workbench.py check results/study.json
```

The evidence binds the exact core source SHA-256. Checking recomputes all six fits, compares structure/strings exactly and numeric values with absolute tolerance 1e-10. Python/NumPy version strings are recorded but not required to match. Other BLAS/version differences can exceed tolerance and should be investigated, not silently blessed.

## Limitations and discussion

- One finite template, 18 tokens, context length 11 and one split. Held-out combinations share grammar and words with training. No independent corpora, significance claim, real-text evaluation or out-of-distribution validation.
- Sampling is unconstrained and can violate the grammar, repeat words or fail to emit EOS. Generation stops at the learned context boundary, without extrapolating position embeddings.
- The task and attention advantage are educational demonstrations, not architectural novelty or evidence of general reasoning. Removing attention changes active capacity as well as context access.
- No GPU measurements or performance comparisons. No training-resume support. No public hosted inference service.

## Tests and attribution

Eleven core test groups include all-parameter finite differences on both small-model variants, a scalar-loop forward oracle, causal-prefix invariance, stable extreme logits, first-step Adam algebra, leakage poisoning, independent metrics/count baselines and fixed-budget training. Four workbench groups cover trained checkpoint round trips for both variants, corruption/schema rejection, bounded parsing, overwrite protection and evidence comparison. CI reruns both suites, the study and the CLI demo.

Mathematical references: [Vaswani et al., Attention Is All You Need](https://arxiv.org/abs/1706.03762), sections 3.1–3.2; [Kingma and Ba, Adam](https://arxiv.org/abs/1412.6980). No third-party implementation, paper text, figures, corpus or assets are incorporated. NumPy is an external [BSD-licensed dependency](https://numpy.org/doc/stable/license.html), not vendored. Code and original generated corpus are MIT licensed; see [LICENSE](LICENSE).
