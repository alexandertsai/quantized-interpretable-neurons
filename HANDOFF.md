# Resume this study

The BF16/FP8 experiments and NVFP4 reference extension are complete. Scientific
code, numerical results, validation, analysis and plots are committed. No task
GPU rentals remain active. No new experiment was started during the final
compute-access check.

## Start here

- `README.md`: findings across the experiments.
- `fp4/README.md`: full 4-bit methods, results and limitations.
- `fp4/protocol.json`: pinned checkpoints, runtime, sample and interventions.
- `fp4/verification.json` and `fp4/preflight*.json`: validation evidence.
- `fp4/analysis.json` and `fp4/results_*.json`: numerical results and intervals.
- `followup/README.md`: earlier causal transfer and inconclusive semantic test.

The result-producing FP4 source commit is
`9510afe1fea0ad49fab1c31e043740ef715371b8`. Results and analysis were saved in
`e0585f1773cb32bf709835974144543556daebbb`.

## Main findings

On 32 matched WikiText articles, retaining the strongest 320 of 4,096 routed
neuron activations adds 3.89% cross-entropy loss in BF16, 3.70% in FP8 and
3.76% in the NVFP4 reference. The FP4-minus-BF16 pruning-overhead difference
has a paired 95% interval of [-0.70, +0.45] percentage points; this pilot does
not resolve a difference. Unmasked FP4 reference loss is 1.86% higher than BF16.

97.64% of BF16-selected identities remain available under FP4 clean routing.
Transferred masks add 0.04760 nats/token of loss, native matched-count masks
add 0.05627, and random matched-count masks add 2.85237. Transferred masks
have greater KL than native masks, so the comparison depends on the metric.

## Important limits

The FP4 checkpoint is the pinned **Red Hat** release, not an official Qwen
quantization. Its released weights are decoded and activation rounding is
emulated using compressed-tensors, with **BF16 matrix multiplication**. Native
FP4 scale handling and accumulation have not been validated. No native speed
or memory saving is established. Some components retain higher precision as
specified by the release. No training or new calibration was performed.

This extension tests sparsity and causal mask transfer. It does not establish
preserved neuron meanings. The earlier Luna semantic test was inconclusive.

## Suggested next experiment

First compare unmasked loss/logits from a supported native NVFP4 runtime against
the saved reference recipe on the identical input sample. Establishing native
parity is more useful than immediately trying lower bit widths. Native MoE
kernels fuse operations, so adapting neuron interventions may require additional
instrumentation. Do not silently substitute weight-only loading for the full
released weight-and-activation recipe.

The original user cap is **$6 cumulative**. Last reported cumulative Vast
charges were **$2.223**, leaving approximately **$3.777** before any settlement
adjustments. Check current billing and active resources before allocating more.
Use existing credits only for other services unless the user authorizes a
different spending arrangement. Temporary jobs need timeouts, result retrieval,
cleanup, and verification that compute is released.

For remote access, use the user's separate `manage-codex-remote` repository,
which was updated to `4545369` during handoff. Follow its current `AGENTS.md`,
setup and provider checks in the new environment. The current environment
verified Vast access with no instances; Colab, Modal and AWS credentials were
not available here. Provider Python CLIs were installed locally, but setup's
`nc` prerequisite remains missing. Local virtual environments and credentials
are not included in this repository.

## Recreate analysis without renting a GPU

```bash
python -m pip install -r requirements-analysis.txt
python fp4/analyze.py
python fp4/plot.py
```

The GPU setup regenerates the public text sample and verifies its saved hash.
Downloaded models, large reference logits, raw passages, credentials and rental
controls are intentionally excluded from Git. See the per-experiment READMEs
before rerunning anything that allocates compute.
