import json,pathlib,statistics,collections
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parent
def read(name):return json.loads((ROOT/name).read_text())
def ci(values):
    a=np.asarray(values,float);rng=np.random.default_rng(42)
    b=a[rng.integers(0,len(a),size=(5000,len(a)))].mean(1)
    return [float(x) for x in np.quantile(b,[.025,.975])]

def analyze():
    bf=read('results_transfer_bf16.json');fp=read('results_transfer_fp8.json')
    transfer=fp['conditions']['transfer'];matched=fp['conditions']['matched_native']
    diff=np.array(transfer['window_losses'])-np.array(matched['window_losses'])
    kd=np.array(transfer['window_kls'])-np.array(matched['window_kls'])
    out={'mask_transfer':{'conditions':fp['conditions'],'bf16_controls':bf['conditions'],
        'mean_retained_neurons':statistics.mean(r['mean_kept'] for r in fp['coverage']),
        'minimum_retained_in_any_token_layer':min(r['min_kept'] for r in fp['coverage']),
        'transfer_minus_matched_native_loss_nats':float(diff.mean()),'loss_difference_95pct_paired_document_bootstrap':ci(diff),
        'transfer_minus_matched_native_kl_nats':float(kd.mean()),'kl_difference_95pct_paired_document_bootstrap':ci(kd)}}
    predictions={}
    for path in ROOT.glob('luna_predictions_*.json'):
        data=json.loads(path.read_text());rows=data['predictions'] if isinstance(data,dict) else data
        for row in rows:
            assert row['id'] not in predictions;assert row['choice'] in ['A','B'];assert 0<=row['p_a']<=1
            predictions[row['id']]=row
    if predictions:
        truth=read('detection_truth.json');assert all(r['case_id'] in predictions for r in truth['rows'])
        metrics={};perpole={}
        for precision in ['bf16','fp8']:
            for kind in ['released','shuffled_description']:
                rows=[r for r in truth['rows'] if r['precision']==precision and r['kind']==kind]
                scores=collections.defaultdict(list)
                for r in rows:scores[r['pid']].append(int(predictions[r['case_id']]['choice']==r['correct']))
                means={p:statistics.mean(s) for p,s in scores.items()};perpole[(precision,kind)]=means
                metrics[precision+'_'+kind]={'pairs':len(rows),'poles':len(means),'micro_accuracy':statistics.mean(s for v in scores.values() for s in v),'macro_accuracy':statistics.mean(means.values()),'macro_95pct_pole_bootstrap':ci(list(means.values())),'per_pole':means}
        common=sorted(set(perpole[('bf16','released')])&set(perpole[('fp8','released')]))
        d=[perpole[('fp8','released')][p]-perpole[('bf16','released')][p] for p in common]
        common_metrics={}
        for precision in ['bf16','fp8']:
            for kind in ['released','shuffled_description']:
                scores=[perpole[(precision,kind)][p] for p in common]
                common_metrics[precision+'_'+kind]={'poles':len(common),'macro_accuracy':statistics.mean(scores),'macro_95pct_pole_bootstrap':ci(scores)}
        pair_lookup={(r['precision'],r['pid'],r['fire_index'],r['control_index']):r for r in truth['rows'] if r['kind']=='released'}
        description_effect={}
        for precision in ['bf16','fp8']:
            delta=collections.defaultdict(list)
            for row in truth['rows']:
                if row['precision']!=precision or row['kind']!='shuffled_description':continue
                r=pair_lookup[(precision,row['pid'],row['fire_index'],row['control_index'])]
                true_correct=int(predictions[r['case_id']]['choice']==r['correct'])
                shuffled_correct=int(predictions[row['case_id']]['choice']==row['correct'])
                delta[row['pid']].append(true_correct-shuffled_correct)
            values=[statistics.mean(v) for v in delta.values()]
            description_effect[precision]={'matched_pairs':sum(len(v) for v in delta.values()),'macro_released_minus_shuffled_accuracy':statistics.mean(values),'95pct_pole_bootstrap':ci(values)}
        fixed=truth['fixed_bf16_pairs'];ranked=[r for r in fixed if r['fp8_ordering_correct'] is not None]
        out['description_detection']={'metrics':metrics,'common_pole_count':len(common),'common_pole_metrics':common_metrics,'matched_description_control_effect':description_effect,'fp8_minus_bf16_macro_accuracy':statistics.mean(d),'accuracy_difference_95pct_paired_pole_bootstrap':ci(d),
            'fixed_bf16_pairs':{'pairs':len(fixed),'original_fire_routed_in_fp8_fraction':statistics.mean(r['fp8_original_fire_routed'] for r in fixed),'original_fire_above_fp8_q97_fraction':statistics.mean(r['fp8_original_fire_still_above_q97'] for r in fixed),'fp8_retains_bf16_pair_activation_order_fraction':statistics.mean(r['fp8_ordering_correct']==r['correct'] for r in ranked),'fp8_ranking_ties':len(fixed)-len(ranked),'judge_accuracy_against_fp8_ordering':statistics.mean(predictions[r['case_id']]['choice']==r['fp8_ordering_correct'] for r in ranked)},
            'eligibility':truth['eligibility'],'judge_model':'gpt-6-luna','unique_judged_pairs':len(predictions)}
        obs={p:np.load(ROOT/f'detection_observations_{p}.npz')['evaluation'] for p in ['bf16','fp8']}
        corrs=[]
        selection_file='descriptions.json' if (ROOT/'descriptions.json').exists() else 'pole_selection.json'
        for i,p in enumerate(read(selection_file)['poles']):
            x=obs['bf16'][...,i].reshape(-1);y=obs['fp8'][...,i].reshape(-1);valid=np.isfinite(x)&np.isfinite(y)
            if valid.sum()>3 and np.std(x[valid])>0 and np.std(y[valid])>0:corrs.append({'pid':p['pid'],'shared_routed_tokens':int(valid.sum()),'pearson':float(np.corrcoef(x[valid],y[valid])[0,1])})
        out['description_detection']['activation_correlations']=corrs
        out['description_detection']['median_activation_pearson_shared_routing']=statistics.median(r['pearson'] for r in corrs)
    (ROOT/'analysis.json').write_text(json.dumps(out,indent=2))
    print('MASK_TRANSFER',{k:v for k,v in out['mask_transfer'].items() if k not in ['conditions','bf16_controls']})
    for name,c in fp['conditions'].items():print(name,c['loss_nats'],c.get('delta_loss_nats'),c['kl_nats'])
    if 'description_detection' in out:
        for k,v in out['description_detection']['metrics'].items():print(k,{q:v[q] for q in ['pairs','poles','macro_accuracy','micro_accuracy']})
        print('DETECTION_DIFFERENCE',out['description_detection']['fp8_minus_bf16_macro_accuracy'],out['description_detection']['accuracy_difference_95pct_paired_pole_bootstrap'])
        print('FIXED_PAIRS',out['description_detection']['fixed_bf16_pairs'])
if __name__=='__main__':analyze()
