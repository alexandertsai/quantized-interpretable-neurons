# BF16/FP8 follow-up experiments

Both experiments use official Qwen3.6-35B-A3B releases, pinned to the revisions
recorded in `base.py`. No training or custom weight quantization. The original
32 WikiText test articles supply 16,352 scored next-token predictions per causal
condition. All GPU rentals were destroyed after data collection.

## 1. Do BF16-selected neurons remain causally useful in FP8?

We record each token/layer's 320 largest absolute router-weighted SwiGLU
activations in an unmasked BF16 pass, as global expert/neuron identities. We
record the same information from unmasked FP8. All selected-neuron interventions
zero the other routed activations before the down projection, without changing
the shared experts or copying BF16 activation values into FP8.

The primary comparison fixes FP8 expert indices and router weights to its own
clean pass. It compares the available BF16-selected neurons against FP8's own
clean top-m neurons and a random-m set. Each token/layer has exactly the same m
under all three conditions, so unavailable experts do not create a budget
confound. This is a diagnostic with frozen routing and static clean masks;
it is not the naturally adaptive top-320 policy in the first pilot.

Mean m is 316.26, or 98.83% of the original 320. The minimum at any token/layer
is zero; average coverage does not imply that every individual case is covered.

| FP8 condition | Added loss (nats/token) | KL from clean FP8 |
| --- | ---: | ---: |
| Available BF16-selected mask | 0.05615 | 0.10485 |
| FP8-selected, same retained count | 0.05887 | 0.10178 |
| Random, same retained count | 2.86356 | 3.02586 |

Transfer minus matched FP8 selection changes loss by -0.00273 nats/token,
with a paired-document bootstrap 95% interval [-0.00655, +0.00118]. It increases
KL by +0.00307, interval [+0.00198, +0.00415]. Thus useful computation transfers
well by this test, while the prediction distribution changes slightly more.
This is not a formal equivalence test or proof of unchanged semantic roles.

All six instrumentation, frozen-routing and keep-all checks produced
bitwise-identical unmasked logits. The adaptive top-320 controls also recover
the first pilot's loss/KL measurements.

## 2. Do released descriptions predict firing across precision?

We sampled 32 released neuron-poles without looking at our activations: 16
positive and 16 negative, stratified across layers 9/14/19/24/30/36. Fixed mode
descriptions came from Goodfire's immutable public release:
https://static.goodfire.ai/neuron-atlas/2026-11-09-082204/

We calibrate signed router-weighted activation percentiles on 64 WikiText
training articles, separate from 32 test articles. Firing examples exceed q97;
controls are below q90 and routed to the same expert. Each pole contributes up
to four firing/control pairs from distinct firing documents. Snippets show the
60 preceding tokens, a marked target token, and eight following tokens.
Some poles lack enough held-out firings: 30 poles and 87 pairs are scored per
precision, with 29 poles common to both. No replacement poles were selected.

Four fresh `gpt-6-luna` subagents judged disjoint blind batches with no parent
history, precision labels, activation values, pole IDs or answer key. Exact
duplicate description/passage pairs are judged once and reused. There are 272
distinct judged pairs and 281 scored rows, including shuffled-description
controls. The judge prompt template, raw responses and numerical answer key are
included. Generated passage inputs and released description text are recreated
locally by the preparation scripts rather than redistributed here.

Main comparison, macro-averaged over the 29 common poles:

| Description condition | BF16 | FP8 |
| --- | ---: | ---: |
| Released description | 62.07% | 60.34% |
| Shuffled description | 53.45% | 48.28% |

The paired difference is -1.72 percentage points for FP8, with a wide 95%
bootstrap interval [-16.67, +14.09]. Both released-description accuracy
intervals include chance. On identical underlying passage pairs, the correct
description's advantage over the shuffled description is also uncertain:
BF16 +5.0 pp [-13.3, +23.3], FP8 +11.7 pp [-8.3, +31.7].

This result is inconclusive about semantic preservation. The paper reports
roughly 95% detection against random text and 89% against same-expert controls
over its reported activation bands. Our same-expert pilot does not reproduce
that high accuracy. English Wikipedia, corpus-relative percentiles, different
band sampling, few examples per pole and a different judge substantially limit
the comparison. The released descriptions cover code, mathematics and other
languages that this corpus may not exercise. Overlap with the original authors'
unknown discovery corpus cannot be ruled out.

The numerical activations are more stable than hard-threshold firing labels:
median Pearson correlation is 0.9825 on jointly routed tokens. For the 87 fixed
BF16 firing/control pairs, FP8 preserves ordering in 85 of 86 non-tied pairs
(98.84%), keeps the original firing routed in 94.25%, and above its own q97 in
80.46%. One pair is tied. These are activation statistics, not evidence that the
descriptions are valid.

Primary scoring uses the explicit forced A/B choice. Of 272 returned probability
fields, 67 disagree with their explicit choice. Raw outputs are preserved;
numeric confidence is not used for scoring or AUROC claims. See
`judge-validation.json`.
An always-A position baseline scores 50.83% and 51.94% macro accuracy for the
separate BF16 and FP8 pole sets. Position-label frequencies and choice bias are
recorded in `position-diagnostic.json` as an additional judge-quality diagnostic.

## Reproduction and files

Use the PyTorch 2.10.0/CUDA 12.8 runtime image, install `setup.sh`, then:

```bash
.venv/bin/python select_descriptions.py
.venv/bin/python prepare_data.py
bash run.sh
```

`followup.py` performs the GPU work. `prepare_judge.py` creates the frozen blind
batches and answer key. Re-run the saved judge prompt with `gpt-6-luna`, using
fresh contexts for each batch, then run `analyze.py` and `plot.py` (NumPy and
Matplotlib are required for the analysis/plotting scripts).

`protocol.json` records the planned protocol and source hashes before judging.
`analysis.json` contains full results and paired bootstrap intervals.
`detection_observations_*.npz` preserves the measured calibration/test
activations. Generated `tokens_*.json` files preserve exact input IDs locally;
the committed `sample_metadata.json` records the evaluation hash and offsets.
The heavyweight model checkpoints and full-vocabulary reference tensors are
not bundled. Account credentials and SSH configuration are also excluded.
