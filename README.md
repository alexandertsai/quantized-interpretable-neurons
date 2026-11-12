# quantized-interpretable-neurons
Follow-up study of Goodfire's neuron interpretability work, examining behavior
across quantization levels. These are small, training-free experiments using
the official BF16 and FP8 releases of Qwen3.6-35B-A3B, plus a reference
experiment using Red Hat’s released NVFP4 checkpoint.

Reference: [Neurons as an Interpretable Basis](https://www.goodfire.com/research/neurons-as-an-interpretable-basis).

## Findings

- **Activation sparsity:** top-320 filtering removes 92.2% of routed neuron
  activations. Loss increased 3.89% in BF16 and 3.70% in FP8 on 32 WikiText test
  articles. Random-320 filtering increased loss by about 135% in both.
- **4-bit reference extension:** top-320 filtering adds 3.76% loss with the
  released NVFP4 recipe, close to the BF16/FP8 results. BF16-selected masks
  remain useful. FP4 values and activation rounding are evaluated using BF16
  matrix multiplications; native FP4 kernels and neuron meanings are untested.
- **Causal mask transfer:** available BF16-selected neurons remained useful in
  FP8. With frozen routing and matched retained counts, added loss was 0.0561
  nats/token for transferred masks, 0.0589 for FP8-selected masks, and 2.8636 for
  random masks. Transferred masks had slightly greater KL divergence.
- **Description prediction:** four fresh, blind Luna subagents judged passage
  pairs from released neuron descriptions. Accuracy was 62.1% for BF16 and 60.3%
  for FP8 across 29 common poles, with wide uncertainty intervals. This test is
  inconclusive about semantic preservation; it does not reproduce the article's
  high detection accuracy.

These findings support sparse computation and causal mask transfer in this
pilot. They do not establish that neuron meanings or reasoning capabilities
are preserved. Corpus, context, threshold and judge differences limit direct
comparison with the original article. The FP4 reference does not establish
native FP4 kernel behavior or deployment savings.

## Repository contents

| Directory | Contents |
| --- | --- |
| [sparsity/](sparsity/) | Initial BF16/FP8 sparsity sweep, methods, numerical results and plots |
| [fp4/](fp4/) | Released NVFP4 reference pilot, weight-only control, matched mask transfer, uncertainty estimates and plots |
| [followup/](followup/) | Matched-budget mask transfer, description detection, blind judge protocol, numerical observations and results |

The scientific scripts are preserved alongside their outputs. Small packaging
changes make shell paths relative and let analysis use public sample metadata
when generated text inputs are absent. GPU intervention and judge-input logic
remain unchanged from the recorded experiments.

Credentials, rental controls, SSH configuration, downloaded checkpoints,
original article text and generated passage/description text are excluded.
Preparation scripts regenerate the public inputs from pinned sources.

## Recreate the saved analysis

Python 3.12 was used. From an environment with NumPy and Matplotlib:

```bash
python -m pip install -r requirements-analysis.txt
python sparsity/analyze.py
python sparsity/plot.py
python followup/analyze.py
python followup/plot.py
python fp4/analyze.py
python fp4/plot.py
```

These commands use the committed numerical results and require no GPU.
See the per-experiment READMEs for methods, validation and limitations.

## Rerun GPU experiments

The recorded runtime used `pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime` and a
single 96 GB RTX Pro 6000. The setup scripts reuse the runtime's PyTorch through
a project-local virtual environment and pin Transformers to the recorded commit.
Package snapshots and exact model revisions are included in each directory.

For the expanded sparsity experiment:

```bash
cd sparsity
bash setup.sh
.venv/bin/python experiment.py --precision bf16 --windows 32 --length 512 --document-sampling
.venv/bin/python experiment.py --precision fp8 --windows 32 --length 512 --document-sampling
```

For the two follow-ups, in a separate directory/environment:

```bash
cd followup
bash setup.sh
.venv/bin/python select_descriptions.py
.venv/bin/python prepare_data.py
bash run.sh
```

The follow-up run emits four blind judge batches and a separate answer key.
Use the saved prompt template with fresh `gpt-6-luna` contexts; do not provide
the answer key or other experiment files to judges. Save responses as
`luna_predictions_0.json` through `luna_predictions_3.json`, then run analysis.
The repository includes the original forced-choice responses for auditing.

Model weights and the large full-vocabulary reference tensors are not stored
in Git. No new GPU rental is needed merely to inspect the saved results.
