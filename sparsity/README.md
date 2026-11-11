# BF16 versus FP8 neuron-sparsity pilot

Training-free partial replication of Goodfire's layerwise top-K intervention:
https://www.goodfire.com/research/neurons-as-an-interpretable-basis#router-weighted-top-k-filtering

## Results

Main sample: 32 distinct WikiText test articles, 512 tokens each, 16,352 scored
next-token predictions per condition. No training. Loss increases are relative
to each precision's own unmasked model.

| Routed activations retained | BF16 loss increase | FP8 loss increase |
| --- | ---: | ---: |
| Top-1,024 / 4,096 | 0.77% | 0.80% |
| Top-512 / 4,096 | 2.10% | 2.18% |
| Top-320 / 4,096 | 3.89% | 3.70% |
| Top-256 / 4,096 | 4.82% | 4.78% |
| Random-320 / 4,096 | 134.91% | 134.50% |

At top-320, KL from the corresponding clean model is 0.16639 nats/token for
BF16 and 0.16815 for FP8. The difference in added loss (FP8 minus BF16) is
-0.00391 nats/token; its paired-document bootstrap 95% interval is
[-0.00974, +0.00188]. This provides no clear evidence of an additional FP8
sparsification penalty on this sample; it is not a formal equivalence test.

Unmasked loss is 2.12845 nats/token for BF16 and 2.13329 for FP8 (+0.23%).
Across the sampled positions, selected-expert sets overlap by 95.18%, and
top-320 neuron sets overlap by 89.86%. Overlap uses the first 32 tokens of each
window and all 40 layers; it does not measure semantic similarity.

All four original-versus-instrumented and keep-all checks are bitwise exact.
All 41 matching checkpoint router tensors (including the auxiliary MTP router)
are also bitwise identical across the official releases. Measured peak allocated
GPU memory is 72.76 GB for BF16 and 39.15 GB for FP8 (decimal units).

The article reports a 2.9% loss increase on its much larger web-text corpus.
Our 3.89% BF16 result supports the qualitative sparsity finding, but does not
reproduce that exact number. Corpus and context differences limit that comparison.
Semantic modes, neuron descriptions, and reasoning capabilities remain untested.

See `analysis.json` for the full measurements, `sparsity-comparison.png` for the
plot, and `initial/` for the separate 2,040-token sanity run.

Run `experiment.py` in the provided PyTorch CUDA environment after `setup.sh`.
The script uses official Qwen BF16 and FP8 checkpoints, pinned to exact revisions.
It retains the top K absolute router-weighted SwiGLU activations among each
token's 8 x 512 routed neurons in each layer. Other routed activations become
zero before the down projection. Shared experts are unchanged. Routing is
allowed to change naturally after each intervention.

## Commands

```bash
bash setup.sh
.venv/bin/python -u experiment.py --precision bf16 --windows 8 --length 256
.venv/bin/python -u experiment.py --precision fp8 --windows 8 --length 256
```

The expanded experiment uses one seeded random 512-token window from each of
32 distinct WikiText test articles (16,352 scored next-token predictions):

```bash
.venv/bin/python -u experiment.py --precision bf16 --windows 32 --length 512 --document-sampling
.venv/bin/python -u experiment.py --precision fp8 --windows 32 --length 512 --document-sampling
python analyze.py
```

Run BF16 first: its clean distributions and activation masks supply the
cross-precision references. Each precision runs in a separate process to free
GPU memory between models. No model training or custom weight quantization.

## Checks and measurements

- Original and instrumented unmasked logits must be bitwise identical.
- Explicit top-4096 (keep everything) logits must be bitwise identical.
- Conditions: unmasked; top-1024, top-512, top-320, top-256; random-320.
- Negative log likelihood in nats/token and KL(clean || intervened).
- FP8 conditions also report KL(BF16 unmasked || FP8 condition).
- Routing and top-320 neuron-set overlap use the first 32 tokens of each window
  at all 40 layers, comparing natural BF16 and FP8 unmasked forward passes.
- FP8 uses the Transformers fine-grained FP8 grouped kernel with dynamic
  activation quantization. BF16 intermediates/nonlinearities are expected;
  this is mixed precision, not a claim that every tensor is FP8.

## Scope and limits

WikiText-2-raw test is a public substitute for the article's unspecified web
corpus. The initial consecutive windows are correlated. The expanded sample
uses distinct documents, but remains a small, single-domain pilot rather than
a representative sample of model capabilities. Exact source token IDs are
generated locally; the committed sample metadata records article titles,
offsets and the evaluation hash. Bootstrap intervals resample the 32
paired documents; they do not account for model, dataset, or kernel choices.
Results do not reproduce benchmark evaluations, reasoning recovery, circuit
discovery, semantic mode counts, or description-based simulation. A loss curve
does not establish that neuron meanings survive quantization. No inference
speedup is claimed: dense kernels still compute the masked activations.

Rental access controls and credentials are excluded from this repository.
