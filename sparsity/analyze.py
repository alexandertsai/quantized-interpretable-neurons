"""Summarize paired pilot results without treating correlated windows as independent documents."""
import json,pathlib,random,statistics
ROOT=pathlib.Path(__file__).resolve().parent
results={p:json.loads((ROOT/f'results_{p}.json').read_text()) for p in ['bf16','fp8']}
summary={'conditions':{},'limitations':['Public WikiText substitute for the original unspecified web corpus.',
 'Short, consecutive windows are correlated and do not establish broad capability retention.',
 'Sparsity and selection stability do not establish semantic interpretability.',
 'Released FP8 configuration changes selected weights and activations together; this does not isolate weight versus activation quantization.']}
bc=results['bf16']['conditions']['unmasked']['loss_nats'];fc=results['fp8']['conditions']['unmasked']['loss_nats']
summary['quantization_only_loss_difference_nats']=fc-bc
summary['quantization_only_relative_loss_increase_percent']=100*(fc/bc-1)
for name,b in results['bf16']['conditions'].items():
    if name not in results['fp8']['conditions']:continue
    f=results['fp8']['conditions'][name]
    assert b['scored_tokens']==f['scored_tokens'],'Results must come from the same evaluation sample'
    row={'bf16':b,'fp8':f,'fp8_minus_bf16_extra_ablation_loss_nats':f['delta_loss_nats']-b['delta_loss_nats']}
    if name!='unmasked':
        bc=results['bf16']['conditions']['unmasked']['window_losses'];fc=results['fp8']['conditions']['unmasked']['window_losses']
        paired=[(fl-fcl)-(bl-bcl) for fl,fcl,bl,bcl in zip(f['window_losses'],fc,b['window_losses'],bc)]
        row['paired_window_extra_ablation_loss_difference_range']=[min(paired),max(paired)]
        metadata_path=ROOT/'tokens.json'
        if not metadata_path.exists():metadata_path=ROOT/'sample_metadata.json'
        tokens=json.loads(metadata_path.read_text())
        if tokens.get('document_sampling'):
            rng=random.Random(42);n=len(paired)
            boot=sorted(sum(paired[rng.randrange(n)] for _ in range(n))/n for _ in range(5000))
            row['paired_document_bootstrap_95pct_ci_extra_ablation_loss_difference']=[boot[125],boot[4874]]
    summary['conditions'][name]=row
    print(f"{name:12s} BF16 loss={b['loss_nats']:.6f} delta={b['delta_loss_nats']:.6f} ({b['relative_loss_increase_percent']:.2f}%) KL={b['kl_to_same_precision_unmasked_nats']:.6f} | FP8 loss={f['loss_nats']:.6f} delta={f['delta_loss_nats']:.6f} ({f['relative_loss_increase_percent']:.2f}%) KL={f['kl_to_same_precision_unmasked_nats']:.6f}")
overlap_files=sorted(ROOT.glob('overlap_*.json'))
if overlap_files:
    records=[r for path in overlap_files for r in json.loads(path.read_text())]
    summary['routing_overlap_fraction']=statistics.mean(r['routing_overlap_fraction'] for r in records)
    summary['top320_overlap_fraction']=statistics.mean(r['top320_overlap_fraction'] for r in records)
    summary['overlap_by_layer']=[{'layer':layer,**{key:statistics.mean(r[key] for r in records if r['layer']==layer) for key in ['routing_overlap_fraction','top320_overlap_fraction']}} for layer in range(40)]
    print('routing overlap',summary['routing_overlap_fraction'],'top320 overlap',summary['top320_overlap_fraction'])
(ROOT/'analysis.json').write_text(json.dumps(summary,indent=2))
