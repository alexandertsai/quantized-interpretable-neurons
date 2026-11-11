"""Matched-budget mask transfer and a blind description-detection pilot."""
import argparse,json,pathlib,time,types,random,re,hashlib,os
import base as B
import numpy as np
import torch
from datasets import load_dataset
ROOT=pathlib.Path(__file__).resolve().parent
POLES=json.loads((ROOT/'descriptions.json').read_text())['poles']
torch.manual_seed(42);torch.set_num_threads(8)
def emit(event,**data):
    row={'time':time.time(),'event':event,**data};print(json.dumps(row),flush=True)
    with (ROOT/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
def save(name,x):(ROOT/name).write_text(json.dumps(x,indent=2))

def selected_mask(global_ids,routing):
    n,k=global_ids.shape
    matches=global_ids[...,None]//512==routing[:,None,:]
    available=matches.any(-1)
    slot=matches.long().argmax(-1)
    index=slot*512+global_ids%512
    index=torch.where(available,index,4096)
    keep=torch.zeros(n,4097,dtype=torch.bool,device=routing.device)
    keep.scatter_(1,index,True)
    return keep[:,:4096]

class Instrument:
    def __init__(self):
        self.mode='none';self.capture=False;self.observe=False;self.fixed=None;self.bf=None;self.own=None;self.states={};self.observed=None;self.window=0;self.coverage=[]
    def install(self,model):
        count=0
        for name,mod in model.named_modules():
            if not name.endswith('.mlp.experts'):continue
            layer=int(name.split('.layers.')[1].split('.')[0]);original=mod._apply_gate
            def pre(mod,args,layer=layer):
                x,idx,rw=args
                if self.fixed is not None:
                    idx=self.fixed[layer]['routing'].to(x.device).long();rw=self.fixed[layer]['weights'].to(x.device)
                mod._pilot_indices=idx;mod._pilot_weights=rw
                return x,idx,rw
            mod.register_forward_pre_hook(pre)
            targets=[(i,p) for i,p in enumerate(POLES) if p['layer']==layer]
            def gate(mod,x,layer=layer,original=original,targets=targets):
                out=original(x)
                if self.mode=='none' and not self.capture and not(self.observe and targets):return out
                idx=mod._pilot_indices;rw=mod._pilot_weights;n,ntop=idx.shape
                _,perm=torch.sort(idx.reshape(-1));natural=torch.empty_like(out);natural[perm]=out;natural=natural.view(n,ntop,512)
                score=(natural.float()*rw.float().unsqueeze(-1)).view(n,4096)
                if self.observe and targets:
                    for column,pole in targets:
                        match=idx==pole['expert'];routed=match.any(-1);slot=match.long().argmax(-1)
                        values=natural[torch.arange(n,device=idx.device),slot,pole['neuron']].float()*rw[torch.arange(n,device=idx.device),slot].float()*pole['sign']
                        values=values.masked_fill(~routed,float('nan'))
                        self.observed[:,column]=values.cpu()
                if self.capture:
                    top=score.abs().topk(320,dim=-1).indices
                    ids=idx.gather(1,top//512)*512+top%512
                    self.states[layer]={'routing':idx.cpu().to(torch.int16),'weights':rw.cpu(),'ids':ids.cpu().to(torch.int32)}
                if self.mode=='none':return out
                if self.mode in ['adaptive320','keep_all']:
                    k=320 if self.mode=='adaptive320' else 4096
                    keep=torch.zeros_like(score,dtype=torch.bool);keep.scatter_(1,score.abs().topk(k,dim=-1).indices,True)
                else:
                    own=self.own[layer]['ids'].to(idx.device).long()
                    if self.mode=='static320':keep=selected_mask(own,idx)
                    else:
                        transferred=selected_mask(self.bf[layer]['ids'].to(idx.device).long(),idx)
                        m=transferred.sum(-1)
                        if self.mode=='transfer':
                            keep=transferred;self.coverage.append({'layer':layer,'window':self.window,'mean_kept':m.float().mean().item(),'min_kept':m.min().item(),'max_kept':m.max().item()})
                        elif self.mode=='matched_native':
                            chosen=own.clone();chosen[torch.arange(320,device=idx.device)[None,:]>=m[:,None]]=-1
                            # Invalid sentinel ids become the non-routed expert -1.
                            keep=selected_mask(chosen,idx)
                        elif self.mode=='matched_random':
                            gen=torch.Generator(device=idx.device);gen.manual_seed(73000+layer+self.window*100)
                            ranks=torch.rand(n,4096,device=idx.device,generator=gen).argsort(-1)
                            pick=ranks[:,:320];keep=torch.zeros_like(score,dtype=torch.bool)
                            keep.scatter_(1,pick,torch.arange(320,device=idx.device)[None,:]<m[:,None])
                        else:raise ValueError(self.mode)
                        assert torch.equal(keep.sum(-1),m)
                return (natural*keep.view(n,8,512)).reshape(-1,512)[perm]
            mod._apply_gate=types.MethodType(gate,mod);count+=1
        assert count==40

def docs_tokens(tokenizer):
    path=ROOT/'tokens_calibration.json'
    if path.exists():return json.loads(path.read_text())
    ds=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split='train',revision='b08601e04326c79dfdd32d625aee71d232d685c3')
    docs=[];current=[];title='preamble'
    for r in ds:
        line=r['text'];heading=line.strip()
        if re.match(r'^= [^=].*[^=] =$',heading):
            if current:docs.append((title,'\n'.join(current)))
            current=[];title=heading
        current.append(line)
    if current:docs.append((title,'\n'.join(current)))
    rng=random.Random(71042);rng.shuffle(docs);ids=[];meta=[]
    eval_titles={r['title'] for r in json.loads((ROOT/'tokens_eval.json').read_text())['windows']}
    for title,text in docs:
        if title in eval_titles:continue
        tokens=tokenizer(text,add_special_tokens=False)['input_ids']
        if len(tokens)<512:continue
        offset=rng.randrange(len(tokens)-512+1);ids.extend(tokens[offset:offset+512]);meta.append({'title':title,'offset':offset})
        if len(meta)==64:break
    assert len(meta)==64
    data={'ids':ids,'windows':meta,'seed':71042,'split':'train','dataset_revision':'b08601e04326c79dfdd32d625aee71d232d685c3'};save('tokens_calibration.json',data);return data

def run(precision):
    model_id=B.BF16 if precision=='bf16' else B.FP8
    tokenizer=B.AutoTokenizer.from_pretrained(B.BF16,revision=B.REVISIONS[B.BF16])
    calibration=docs_tokens(tokenizer)
    evaluation=json.loads((ROOT/'tokens_eval.json').read_text())
    env={'precision':precision,'model':model_id,'revision':B.REVISIONS[model_id],'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'calibration_documents':64,'evaluation_documents':32,'tokens_per_document':512,'released_poles':len(POLES)}
    save('environment_'+precision+'.json',env);emit('load_start',**env)
    model=B.AutoModelForImageTextToText.from_pretrained(model_id,revision=B.REVISIONS[model_id],dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',experts_implementation='grouped_mm').eval()
    emit('load_done',precision=precision,allocated_gb=torch.cuda.memory_allocated()/1e9)
    instrument=Instrument()
    with torch.inference_mode():
        first=torch.tensor([evaluation['ids'][:512]],device='cuda')
        before=B.logits(model,first);instrument.install(model);after=B.logits(model,first)
        assert torch.equal(before,after);emit('instrumentation_noop',precision=precision,bitwise_equal=True)
        del before,after
        instrument.capture=True;instrument.observe=True;eval_obs=[];clean_losses=[]
        for i in range(32):
            instrument.observed=torch.full((512,len(POLES)),float('nan'));instrument.states={}
            ids=torch.tensor([evaluation['ids'][i*512:(i+1)*512]],device='cuda')
            out=B.logits(model,ids);ref=out.log_softmax(-1).cpu()
            loss,_=B.measure(out,ids[0,1:],ref);clean_losses.append(loss/511)
            torch.save({'layers':instrument.states,'logp':ref},ROOT/f'{precision}_clean_{i}.pt')
            eval_obs.append(instrument.observed.numpy());emit('clean_window',precision=precision,window=i,loss=loss/511)
            del out,ref
        instrument.capture=False;instrument.observe=False
        own=torch.load(ROOT/f'{precision}_clean_0.pt',weights_only=True)
        instrument.fixed=own['layers'];instrument.own=own['layers']
        frozen=B.logits(model,first);instrument.fixed=None;natural=B.logits(model,first)
        assert torch.equal(frozen,natural);emit('routing_noop',precision=precision,bitwise_equal=True)
        instrument.mode='keep_all';kept=B.logits(model,first);assert torch.equal(kept,natural)
        emit('keep_all_noop',precision=precision,bitwise_equal=True);del frozen,natural,kept,own
        conditions=['adaptive320','static320'] if precision=='bf16' else ['adaptive320','static320','transfer','matched_native','matched_random']
        results={'precision':precision,'conditions':{'unmasked':{'loss_nats':sum(clean_losses)/32,'window_losses':clean_losses,'kl_nats':0,'window_kls':[0.0]*32,'scored_tokens':16352}}}
        for mode in conditions:
            instrument.mode=mode;losses=[];kls=[]
            for i in range(32):
                own=torch.load(ROOT/f'{precision}_clean_{i}.pt',weights_only=True)
                instrument.own=own['layers'];instrument.fixed=None if mode=='adaptive320' else own['layers'];instrument.window=i
                if precision=='fp8' and mode in ['transfer','matched_native','matched_random']:
                    other=torch.load(ROOT/f'bf16_clean_{i}.pt',weights_only=True);instrument.bf=other['layers'];del other
                ids=torch.tensor([evaluation['ids'][i*512:(i+1)*512]],device='cuda')
                out=B.logits(model,ids);loss,kl=B.measure(out,ids[0,1:],own['logp'])
                losses.append(loss/511);kls.append(kl/511);del out,own
            row={'loss_nats':sum(losses)/32,'window_losses':losses,'kl_nats':sum(kls)/32,'window_kls':kls,'scored_tokens':16352}
            row['delta_loss_nats']=row['loss_nats']-results['conditions']['unmasked']['loss_nats']
            row['relative_loss_increase_percent']=100*(row['loss_nats']/results['conditions']['unmasked']['loss_nats']-1)
            results['conditions'][mode]=row;save('results_transfer_'+precision+'.json',results);emit('condition_done',precision=precision,condition=mode,loss=row['loss_nats'],kl=row['kl_nats'])
        if precision=='fp8':results['coverage']=instrument.coverage
        results['peak_allocated_gb']=torch.cuda.max_memory_allocated()/1e9;save('results_transfer_'+precision+'.json',results)
        # Calibration is disjoint from all evaluation articles. Skip the LM head.
        instrument.mode='none';instrument.fixed=None;instrument.observe=True;cal_obs=[]
        for i in range(64):
            instrument.observed=torch.full((512,len(POLES)),float('nan'))
            ids=torch.tensor([calibration['ids'][i*512:(i+1)*512]],device='cuda')
            model.model.language_model(input_ids=ids,use_cache=False)
            cal_obs.append(instrument.observed.numpy())
            if i%16==0:emit('calibration_progress',precision=precision,window=i)
        np.savez_compressed(ROOT/f'detection_observations_{precision}.npz',calibration=np.stack(cal_obs),evaluation=np.stack(eval_obs))
        emit('precision_done',precision=precision)
    if precision=='fp8':
        from prepare_judge import prepare
        prepare()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--precision',choices=['bf16','fp8'],required=True);run(p.parse_args().precision)
