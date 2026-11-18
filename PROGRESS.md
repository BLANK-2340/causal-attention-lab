# Task-owned continuity

## Registration

Fourth November project: https://github.com/BLANK-2340/causal-attention-lab, public, created November 18, 2026 after checking exact name availability, owner BLANK-2340 (105676435) and visibility. Repository ID 1462709133. The initial CPU implementation is commit 7370e2c1cd923d0c999b1bb04c36f10d84cdaca8.

## November 18, second session

Added strict inference-only JSON checkpoints, generation/attention inspection CLI, source-bound six-run evidence and recomputation, four workbench test groups, pinned NumPy dependency, README architecture/protocol/results/limitations, separate MIT license and CI. The original core model file remains byte-for-byte unchanged. Its earlier module-docstring pending-documentation status is historical, superseded by this record.

Local acceptance: 11 warning-strict core groups and four workbench groups pass. Both trained model variants round-trip with exact weights, held-out metrics and generation. Parser rejects oversized, duplicate-key, invalid-schema/type/shape, nonfinite and damaged/checksum-mismatched data. Existing checkpoint output is refused without byte changes. The full six-run study reproduces the initial fixed-split evidence. CLI generation and inspection execute actual CPU inference. See README for measured results and source/licensing records.

Publication acceptance still requires final diff/clean-archive reruns and remote tree verification; remote CI status must be checked after publishing. This is a bounded educational synthetic LM implementation, not a competitive model. No hosted browser demo, natural-text evaluation, optimizer-state persistence or GPU validation is claimed. The first-three projects' required public demos were already verified November 17.

Next milestone after publication/CI verification: create/register the fifth NEW public CUDA-fusion project under BLANK-2340, checking exact owner/name availability. Implement independent CPU references and numerical/boundary tests plus one coherent fused-kernel benchmark harness with warmup, repetitions, correctness, compiler/hardware metadata and explicit timing scope. Actual CUDA compilation, GPU correctness and performance measurements require execution on compatible free hardware; otherwise mark them pending. Do not rebuild prior projects or repurpose an old repository. All-five completion remains outstanding until the fifth project and its exact runtime limitations are reported.
