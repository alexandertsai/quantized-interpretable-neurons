"""Document-paired uncertainty estimates for the NVFP4 reference pilot."""
import json,pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent
def load(name):return json.loads((ROOT/name).read_text())
def main():
    results={p:load('results_'+p+'.json') for p in ['bf16','fp4_w4a4','fp4_w4a16']}
    n=len(results['bf16']['conditions']['unmasked']['window_losses'])
    rng=np.random.default_rng(42);indices=rng.integers(0,n,size=(10000,n))
    def summary(x):
        x=np.asarray(x);boot=x[indices].mean(1)
        return {'mean':float(x.mean()),'document_bootstrap_95ci':[float(v) for v in np.quantile(boot,[.025,.975])]}
    out={'documents':n,'tokens_per_document':512,'bootstrap_resamples':10000,'conditions':{}}
    for p,r in results.items():
        base=np.array(r['conditions']['unmasked']['window_losses']);conditions={}
        for c,row in r['conditions'].items():
            loss=np.array(row['window_losses']);ratio=100*((loss[indices].mean(1)/base[indices].mean(1))-1)
            conditions[c]={'loss_nats':summary(loss),'delta_loss_nats':summary(loss-base),'relative_loss_increase_percent':{'estimate':100*(loss.mean()/base.mean()-1),'document_bootstrap_95ci':[float(v) for v in np.quantile(ratio,[.025,.975])]},'kl_nats':summary(row['window_kls'])}
        out['conditions'][p]=conditions
    fp4=results['fp4_w4a4']['conditions'];bf16=results['bf16']['conditions'];w16=results['fp4_w4a16']['conditions']
    out['fp4_vs_bf16_unmasked_delta_loss_nats']=summary(np.array(fp4['unmasked']['window_losses'])-bf16['unmasked']['window_losses'])
    fbase=np.array(fp4['unmasked']['window_losses']);bbase=np.array(bf16['unmasked']['window_losses'])
    ftop=np.array(fp4['adaptive320']['window_losses']);btop=np.array(bf16['adaptive320']['window_losses'])
    diff=100*(ftop[indices].mean(1)/fbase[indices].mean(1)-btop[indices].mean(1)/bbase[indices].mean(1))
    out['fp4_minus_bf16_pruning_overhead_percentage_points']={'estimate':float(100*(ftop.mean()/fbase.mean()-btop.mean()/bbase.mean())),'document_bootstrap_95ci':[float(v) for v in np.quantile(diff,[.025,.975])]}
    out['activation_rounding_unmasked_delta_loss_nats']=summary(np.array(fp4['unmasked']['window_losses'])-w16['unmasked']['window_losses'])
    for metric in ['window_losses','window_kls']:
        out['transfer_minus_native_'+metric]=summary(np.array(fp4['transfer'][metric])-fp4['matched_native'][metric])
    coverage=results['fp4_w4a4']['coverage']
    out['transferred_mean_kept']=float(np.mean([c['mean_kept'] for c in coverage]));out['transferred_available_percent']=100*out['transferred_mean_kept']/320
    (ROOT/'analysis.json').write_text(json.dumps(out,indent=2))
    for p,rows in out['conditions'].items():
        for c,r in rows.items():print(p,c,round(r['loss_nats']['mean'],6),'delta%',round(r['relative_loss_increase_percent']['estimate'],3),'KL',round(r['kl_nats']['mean'],6))
    print('transferred_available_percent',out['transferred_available_percent'])
if __name__=='__main__':main()
