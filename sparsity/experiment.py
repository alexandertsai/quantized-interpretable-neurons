"""Small, training-free partial replication of Goodfire's router-weighted top-K."""
import argparse, gc, hashlib, importlib.metadata, json, os, pathlib, platform, random, re, subprocess, time, types
os.environ.setdefault('TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR','1')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS','1')
import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoTokenizer
from datasets import load_dataset

ROOT=pathlib.Path(__file__).resolve().parent
BF16='Qwen/Qwen3.6-35B-A3B'
FP8=BF16+'-FP8'
REVISIONS={BF16:'345d467b117d1fd5eda9b879a5988de9ae5f47a7',FP8:'c653bba46dfa06e9658dcc0b0dc87fe4685f14a2'}
torch.manual_seed(42);torch.set_num_threads(8)

def emit(event,**kwargs):
    row={'time':time.time(),'event':event,**kwargs}
    print(json.dumps(row),flush=True)
    with (ROOT/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')

def save_json(name,data):
    temp=ROOT/(name+'.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(ROOT/name)

class Masker:
    def __init__(self):self.k=None;self.random=False;self.capture=False;self.records={};self.calls=0
    def install(self,model):
        count=0
        for name,module in model.named_modules():
            if name.endswith('.mlp.experts'):
                assert hasattr(module,'_apply_gate'),(name,type(module))
                original=module._apply_gate
                layer=int(name.split('.layers.')[1].split('.')[0])
                def pre(mod,args,layer=layer):
                    mod._pilot_indices=args[1];mod._pilot_weights=args[2]
                module.register_forward_pre_hook(pre)
                def gate(mod,inputs,original=original,layer=layer):
                    out=original(inputs)
                    if self.k is None and not self.capture:return out
                    ids=mod._pilot_indices;rw=mod._pilot_weights
                    n,ntop=ids.shape;dim=out.shape[-1]
                    # grouped_mm orders token-expert pairs by expert id.
                    _,perm=torch.sort(ids.reshape(-1))
                    natural=torch.empty_like(out);natural[perm]=out
                    natural=natural.view(n,ntop,dim)
                    scores=(natural.float()*rw.float().unsqueeze(-1)).abs().view(n,-1)
                    if self.capture:
                        take=min(n,32)
                        strongest=scores[:take].topk(320,dim=-1).indices
                        expert=ids[:take].gather(1,strongest//dim)
                        global_ids=expert*dim+strongest%dim
                        self.records[layer]={'routing':ids[:take].cpu(),'strong_ids':global_ids.cpu(),
                            'strong_values':(natural[:take].float()*rw[:take].float().unsqueeze(-1)).reshape(take,-1).gather(1,strongest).cpu()}
                    if self.k is None:return out
                    if self.random:
                        gen=torch.Generator(device=out.device);gen.manual_seed(42000+layer+self.calls*100)
                        selected=torch.rand(scores.shape,device=out.device,generator=gen).topk(self.k,dim=-1).indices
                    else:selected=scores.topk(self.k,dim=-1).indices
                    keep=torch.zeros_like(scores,dtype=torch.bool);keep.scatter_(1,selected,True)
                    natural=natural*keep.view(n,ntop,dim)
                    return natural.view(-1,dim)[perm]
                module._apply_gate=types.MethodType(gate,module);count+=1
        assert count==40,count
        emit('instrumentation_installed',layers=count)

def logits(model,tokens):
    return model(input_ids=tokens,use_cache=False).logits[0,:-1].float()

def measure(logit,targets,ref):
    loss=F.cross_entropy(logit,targets,reduction='sum').item();kl=0.0
    for start in range(0,logit.shape[0],32):
        lp=logit[start:start+32].log_softmax(-1)
        rp=ref[start:start+32].to(logit.device)
        kl+=(rp.exp()*(rp-lp)).sum().item()
    return loss,kl

def run(args):
    model_id=BF16 if args.precision=='bf16' else FP8
    env={'python':platform.python_version(),'gpu':torch.cuda.get_device_name(),'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
         'packages':{k:importlib.metadata.version(k) for k in ['torch','transformers','triton','kernels','datasets','huggingface_hub']},
         'revisions':REVISIONS,'dataset':'Salesforce/wikitext:wikitext-2-raw-v1:test','dataset_revision':'b08601e04326c79dfdd32d625aee71d232d685c3','seed':42,'windows':args.windows,'sequence_length':args.length,
         'document_sampling':args.document_sampling,
         'experts_implementation':'grouped_mm','attention':'sdpa','deepgemm_linear_disabled':True,
         'mask':'router-weighted absolute SwiGLU output, top-K among 8x512, zero before down projection; shared experts untouched'}
    save_json('environment.json',env);emit('environment',**env)
    tokenizer=AutoTokenizer.from_pretrained(BF16,revision=REVISIONS[BF16])
    cached=json.loads((ROOT/'tokens.json').read_text()) if (ROOT/'tokens.json').exists() else {}
    if cached.get('document_sampling',False)==args.document_sampling and len(cached.get('ids',[]))>=args.windows*args.length:
        all_ids=cached['ids']
    else:
        ds=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split='test',revision='b08601e04326c79dfdd32d625aee71d232d685c3')
        metadata=[]
        if args.document_sampling:
            docs=[];current=[];title='preamble'
            for r in ds:
                line=r['text'];heading=line.strip()
                if re.match(r'^= [^=].*[^=] =$',heading):
                    if current:docs.append((title,'\n'.join(current)))
                    current=[];title=heading
                current.append(line)
            if current:docs.append((title,'\n'.join(current)))
            rng=random.Random(42);rng.shuffle(docs);all_ids=[]
            for title,text in docs:
                ids=tokenizer(text,add_special_tokens=False)['input_ids']
                if len(ids)<args.length:continue
                offset=rng.randint(0,len(ids)-args.length)
                all_ids.extend(ids[offset:offset+args.length]);metadata.append({'title':title,'offset':offset,'document_tokens':len(ids)})
                if len(metadata)==args.windows:break
            assert len(metadata)==args.windows,('Too few documents',len(metadata))
        else:
            text='\n\n'.join(r['text'] for r in ds if r['text'].strip())
            all_ids=tokenizer(text,add_special_tokens=False)['input_ids'][:args.windows*args.length]
        save_json('tokens.json',{'ids':all_ids,'sha256':hashlib.sha256(json.dumps(all_ids).encode()).hexdigest(),'document_sampling':args.document_sampling,'windows':metadata})
    assert len(all_ids)>=args.windows*args.length
    emit('load_start',precision=args.precision)
    model=AutoModelForImageTextToText.from_pretrained(model_id,revision=REVISIONS[model_id],dtype=torch.bfloat16,
        device_map={'':'cuda:0'},attn_implementation='sdpa',experts_implementation='grouped_mm').eval()
    dtype_counts={}
    for parameter in model.parameters():dtype_counts[str(parameter.dtype)]=dtype_counts.get(str(parameter.dtype),0)+parameter.numel()
    emit('load_done',precision=args.precision,allocated_gb=torch.cuda.memory_allocated()/1e9,
         expert_class=type(model.model.language_model.layers[0].mlp.experts).__name__,parameter_counts_by_dtype=dtype_counts)
    # Validate hooks against the same runtime's unmodified grouped_mm model.
    ids=torch.tensor([all_ids[:args.length]],device='cuda')
    with torch.inference_mode():
        before=logits(model,ids);torch.cuda.synchronize()
        masker=Masker();masker.install(model)
        after=logits(model,ids)
        diff=(before-after).abs().max().item()
        emit('noop_check',precision=args.precision,max_abs_logit_difference=diff,bitwise_equal=torch.equal(before,after))
        assert torch.equal(before,after),'Instrumentation altered unmasked predictions'
        masker.k=4096
        all_kept=logits(model,ids)
        emit('all_kept_check',precision=args.precision,max_abs_logit_difference=(after-all_kept).abs().max().item(),bitwise_equal=torch.equal(after,all_kept))
        assert torch.equal(after,all_kept),'Keep-all mask changed predictions'
        masker.k=None
        del all_kept
        del before,after
        conditions=[('unmasked',None,False),('top1024',1024,False),('top512',512,False),('top320',320,False),('top256',256,False),('random320',320,True)]
        results={'precision':args.precision,'model':model_id,'revision':REVISIONS[model_id],'noop_max_difference':diff,'conditions':{}}
        for label,k,is_random in conditions:
            masker.k=k;masker.random=is_random
            losses=[];kls=[];cross_kls=[];timings=[]
            for i in range(args.windows):
                t=time.time();masker.calls=i;masker.capture=(label=='unmasked');masker.records={}
                ids=torch.tensor([all_ids[i*args.length:(i+1)*args.length]],device='cuda')
                out=logits(model,ids);torch.cuda.synchronize()
                path=ROOT/f'clean_{args.precision}_{i}.pt'
                if label=='unmasked':
                    ref=out.log_softmax(-1).cpu();torch.save(ref,path)
                    if args.precision=='bf16':torch.save(masker.records,ROOT/f'masks_{i}.pt')
                    else:
                        base=torch.load(ROOT/f'masks_{i}.pt',weights_only=True)
                        comparisons=[]
                        for layer,r in masker.records.items():
                            b=base[layer]
                            routes=(r['routing'].unsqueeze(-1)==b['routing'].unsqueeze(-2)).any(-1).float().mean().item()
                            overlap=(r['strong_ids'].unsqueeze(-1)==b['strong_ids'].unsqueeze(-2)).any(-1).float().mean().item()
                            comparisons.append({'layer':layer,'routing_overlap_fraction':routes,'top320_overlap_fraction':overlap})
                        save_json(f'overlap_{i}.json',comparisons)
                        emit('precision_overlap',window=i,routing_overlap_fraction=sum(r['routing_overlap_fraction'] for r in comparisons)/40,
                             top320_overlap_fraction=sum(r['top320_overlap_fraction'] for r in comparisons)/40)
                else:ref=torch.load(path,weights_only=True)
                loss,kl=measure(out,ids[0,1:],ref)
                if args.precision=='fp8':
                    bfref=torch.load(ROOT/f'clean_bf16_{i}.pt',weights_only=True)
                    _,cross=measure(out,ids[0,1:],bfref);cross_kls.append(cross/(args.length-1))
                losses.append(loss/(args.length-1));kls.append(kl/(args.length-1));timings.append(time.time()-t)
                emit('window_done',precision=args.precision,condition=label,window=i,loss_nats=losses[-1],kl_nats=kls[-1],seconds=timings[-1])
                del out,ref
            row={'loss_nats':sum(losses)/len(losses),'kl_to_same_precision_unmasked_nats':sum(kls)/len(kls),
                 'window_losses':losses,'window_kls':kls,'seconds':sum(timings),'scored_tokens':args.windows*(args.length-1)}
            if cross_kls:row['kl_from_bf16_unmasked_nats']=sum(cross_kls)/len(cross_kls)
            clean=results['conditions'].get('unmasked',row)['loss_nats']
            row['delta_loss_nats']=row['loss_nats']-clean;row['relative_loss_increase_percent']=100*(row['loss_nats']/clean-1)
            results['conditions'][label]=row
            save_json('results_'+args.precision+'.json',results)
            emit('condition_done',precision=args.precision,condition=label,**row)
        results['peak_allocated_gb']=torch.cuda.max_memory_allocated()/1e9
        save_json('results_'+args.precision+'.json',results)
        emit('precision_done',precision=args.precision,peak_allocated_gb=results['peak_allocated_gb'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--precision',choices=['bf16','fp8'],required=True);p.add_argument('--windows',type=int,default=8);p.add_argument('--length',type=int,default=256);p.add_argument('--document-sampling',action='store_true')
    run(p.parse_args())
