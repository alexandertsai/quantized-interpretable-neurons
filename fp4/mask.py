import torch, types
POLES=[]


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
