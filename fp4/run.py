"""Matched text-only BF16 / released NVFP4-reference sparsity pilot."""
import argparse,json,time,pathlib,types,importlib.metadata,zlib,base64,hashlib
import torch
import base as B
from mask import Instrument
import nvfp4_reference as N
ROOT=pathlib.Path(__file__).resolve().parent
def save(name,data):
    p=ROOT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(data,indent=2));t.replace(p)
def emit(event,**kw):
    row={'time':time.time(),'event':event,**kw};print(json.dumps(row),flush=True)
    with (ROOT/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')

def preflight():
    from compressed_tensors.quantization import QuantizationStatus
    from compressed_tensors.quantization.lifecycle.forward import forward_quantize
    from compressed_tensors.compressors.nvfp4 import NVFP4PackedCompressor
    ref=N.Reference(json.loads((ROOT/'RedHatAI_config.json').read_text()))
    x=torch.randn(8,512,device='cuda',dtype=torch.bfloat16);x[0]=0
    gs=torch.tensor([131,776,131,776,5,20,100,400],device='cuda',dtype=torch.float32)[:,None]
    actual=ref.qdq(x,gs);expected=[]
    for i in range(len(x)):
        m=torch.nn.Identity();m.input_global_scale=gs[i];m.quantization_status=QuantizationStatus.FROZEN
        expected.append(forward_quantize(m,x[i:i+1],'input',ref.args))
    expected=torch.cat(expected);difference=(actual-expected).abs().max().item()
    # CPU preflight additionally used an actual released expert. Here validate
    # GPU decoding on random packed values with valid FP8 scales.
    unit={'weight_packed':torch.randint(0,256,(512,1024),device='cuda',dtype=torch.uint8),
          'weight_scale':torch.ones(512,128,device='cuda').to(torch.float8_e4m3fn),
          'weight_global_scale':torch.tensor([31104.],device='cuda')}
    expected_w=NVFP4PackedCompressor.decompress(unit,ref.scheme)['weight']
    actual_w=ref.decode(unit['weight_packed'],unit['weight_scale'],unit['weight_global_scale'])
    row={'activation_qdq_max_abs_difference_to_official':difference,'activation_qdq_bitwise_equal':torch.equal(actual,expected),'weight_decompression_bitwise_equal':torch.equal(actual_w,expected_w),'zero_row_finite':bool(torch.isfinite(actual[0]).all())}
    save('preflight_gpu.json',row);emit('preflight',**row)
    assert row['activation_qdq_bitwise_equal'] and row['weight_decompression_bitwise_equal'] and row['zero_row_finite']

def run(precision,windows):
    tokens=json.loads((ROOT/'tokens.json').read_text())['ids']
    env={'precision':precision,'gpu':torch.cuda.get_device_name(),'packages':{k:importlib.metadata.version(k) for k in ['torch','transformers','compressed-tensors','kernels']},'windows':windows,'sequence_length':512,'scored_tokens':windows*511,'bf16_revision':N.BASE_REV,'nvfp4_model':N.MODEL,'nvfp4_revision':N.REVISION,'matrix_multiplication_dtype':'bfloat16','nvfp4_mode':'reference weight decode plus activation quantize/dequantize; not native FP4 kernels','shared_experts_pruned':False}
    save('environment_'+precision+'.json',env);emit('load_start',**env)
    if precision=='bf16':
        model=B.AutoModelForImageTextToText.from_pretrained(B.BF16,revision=N.BASE_REV,dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='sdpa',experts_implementation='grouped_mm').eval();ref=None
    else:model,ref=N.load(emit=emit)
    emit('load_done',precision=precision,allocated_gb=torch.cuda.memory_allocated()/1e9)
    ins=Instrument()
    first=torch.tensor([tokens[:512]],device='cuda')
    with torch.inference_mode():
        if precision=='bf16':
            # Verify our reference MoE indexing/reduction against the pinned HF
            # implementation with activation quantization disabled.
            before=B.logits(model,first);r=N.Reference(json.loads((ROOT/'RedHatAI_config.json').read_text()));r.enabled=False;original=[]
            for name,m in model.named_modules():
                if name.endswith('.mlp.experts'):
                    original.append((m,m.forward));m.fp4_gate_global=torch.ones(m.num_experts,device='cuda');m.fp4_down_global=m.fp4_gate_global;m.fp4_gate_up_equal=True
                    def forward(mod,x,idx,rw):return r.experts_forward(mod,x,idx,rw)
                    m.forward=types.MethodType(forward,m)
            after=B.logits(model,first);ok=torch.equal(before,after)
            save('preflight_reference_moe.json',{'bf16_custom_moe_bitwise_equal':ok,'max_abs_logit_difference':(before-after).abs().max().item()});assert ok
            for m,forward in original:m.forward=forward
            del before,after,r,original
        before=B.logits(model,first);ins.install(model);after=B.logits(model,first);assert torch.equal(before,after)
        ins.mode='keep_all';kept=B.logits(model,first);assert torch.equal(after,kept);ins.mode='none'
        emit('noop_checks',precision=precision,instrumentation_bitwise_equal=True,keep_all_bitwise_equal=True);del before,after,kept
        # Store full clean routing/top320 IDs, but not full vocabulary logits.
        # The latter stay as scratch files on the rental and are not published.
        def clean(label):
            ins.mode='none';ins.fixed=None;ins.capture=True;losses=[]
            for i in range(windows):
                ins.states={};ids=torch.tensor([tokens[i*512:(i+1)*512]],device='cuda');out=B.logits(model,ids)
                lp=out.log_softmax(-1).cpu();loss,_=B.measure(out,ids[0,1:],lp);losses.append(loss/511)
                torch.save({'layers':ins.states,'logp':lp},ROOT/f'{label}_clean_{i}.pt');del out,lp
            ins.capture=False
            return {'loss_nats':sum(losses)/windows,'window_losses':losses,'kl_nats':0,'window_kls':[0.]*windows,'scored_tokens':windows*511}
        results={'precision':precision,'conditions':{}}
        labels=[precision] if precision=='bf16' else ['fp4_w4a4','fp4_w4a16']
        for label in labels:
            if ref:ref.enabled=label=='fp4_w4a4'
            result={'precision':label,'conditions':{'unmasked':clean(label)}}
            save('results_'+label+'.json',result);emit('condition_done',precision=label,condition='unmasked',loss=result['conditions']['unmasked']['loss_nats'])
            conditions=['adaptive320'] if label!='fp4_w4a4' else ['adaptive320','transfer','matched_native','matched_random']
            for mode in conditions:
                ins.mode=mode;losses=[];kls=[]
                for i in range(windows):
                    own=torch.load(ROOT/f'{label}_clean_{i}.pt',weights_only=True);ins.own=own['layers'];ins.fixed=None if mode=='adaptive320' else own['layers'];ins.window=i
                    if mode in ['transfer','matched_native','matched_random']:
                        other=torch.load(ROOT/f'bf16_clean_{i}.pt',weights_only=True);ins.bf=other['layers'];del other
                    ids=torch.tensor([tokens[i*512:(i+1)*512]],device='cuda');out=B.logits(model,ids)
                    loss,kl=B.measure(out,ids[0,1:],own['logp']);losses.append(loss/511);kls.append(kl/511);del out,own
                row={'loss_nats':sum(losses)/windows,'window_losses':losses,'kl_nats':sum(kls)/windows,'window_kls':kls,'scored_tokens':windows*511}
                row['delta_loss_nats']=row['loss_nats']-result['conditions']['unmasked']['loss_nats'];row['relative_loss_increase_percent']=100*row['delta_loss_nats']/result['conditions']['unmasked']['loss_nats']
                result['conditions'][mode]=row;save('results_'+label+'.json',result);emit('condition_done',precision=label,condition=mode,loss=row['loss_nats'],kl=row['kl_nats'])
            if label=='fp4_w4a4':result['coverage']=ins.coverage
            save('results_'+label+'.json',result)
            # Vast's authenticated logs API is also a result transport. No
            # credentials, model weights or raw text are included in this bundle.
            bundle={p.name:json.loads(p.read_text()) for pattern in ['results_*.json','preflight*.json','environment*.json'] for p in ROOT.glob(pattern)}
            encoded=base64.b64encode(zlib.compress(json.dumps(bundle).encode())).decode()
            digest=hashlib.sha256(encoded.encode()).hexdigest()[:16]
            chunks=[encoded[j:j+400] for j in range(0,len(encoded),400)]
            for j,chunk in enumerate(chunks):print(f'RESULT_CHUNK {digest} {j} {len(chunks)} {chunk}',flush=True)
        emit('precision_done',precision=precision)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('precision',choices=['preflight','bf16','fp4']);p.add_argument('--windows',type=int,default=32);a=p.parse_args()
    if a.precision=='preflight':preflight()
    else:run(a.precision,a.windows)
