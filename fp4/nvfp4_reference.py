"""Decode a released NVFP4 checkpoint and emulate its activation quantization.

This is W4A4 quantize/dequantize reference execution, with BF16 GEMMs, not a
native FP4 kernel or a speed/memory benchmark. No calibration or fitting occurs.
"""
import json, types
import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForImageTextToText
from transformers.integrations.moe import _grouped_mm
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from compressed_tensors.quantization.lifecycle.forward import dequantize, fake_quantize
from compressed_tensors.quantization.utils import compute_dynamic_scales_and_zp
from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8

MODEL='RedHatAI/Qwen3.6-35B-A3B-NVFP4'
REVISION='e94a930c8e9b8a7d3763e449289c261c9c7d3e5e'
BASE='Qwen/Qwen3.6-35B-A3B'
BASE_REV='345d467b117d1fd5eda9b879a5988de9ae5f47a7'

class Reference:
    def __init__(self, config):
        self.enabled=True
        self.scheme=QuantizationScheme.model_validate(config['quantization_config']['config_groups']['group_0'])
        self.args=self.scheme.input_activations
        self.identity=torch.nn.Identity()

    def qdq(self, x, global_scale):
        if not self.enabled:return x
        shape=x.shape
        x=x.reshape(-1,shape[-1])
        scale,zp=compute_dynamic_scales_and_zp(x,self.args,self.identity,global_scale=global_scale)
        # Fold global scale into local scales, supporting a distinct global per
        # selected expert row. CT's GPU quantizer otherwise assumes one scalar.
        out=fake_quantize(x,scale/global_scale,zp,self.args,global_scale=None)
        return out.reshape(shape)

    def decode(self, packed, scale, global_scale):
        rows=packed.numel()//packed.shape[-1]
        k=packed.shape[-1]*2
        raw=unpack_fp4_from_uint8(packed.reshape(rows,k//2),rows,k)
        # Official CT decompression casts local scale to the unpacked BF16 dtype.
        effective=scale.reshape(rows,k//16).to(raw.dtype)/global_scale
        return dequantize(raw,effective,args=self.scheme.weights,global_scale=None,dtype=raw.dtype)

    def experts_forward(self, mod, x, idx, rw):
        n,k=idx.shape
        expert,perm=torch.sort(idx.reshape(-1))
        selected=x[perm//k]
        counts=torch.histc(expert.int(),bins=mod.num_experts,min=0,max=mod.num_experts-1)
        offsets=counts.cumsum(0,dtype=torch.int32)
        q=self.qdq(selected,mod.fp4_gate_global[expert,None])
        if mod.fp4_gate_up_equal:
            gu=_grouped_mm(q,mod.gate_up_proj.transpose(-2,-1),offsets)
        else:
            mid=mod.intermediate_dim
            g=_grouped_mm(q,mod.gate_up_proj[:,:mid,:].transpose(-2,-1),offsets)
            u=_grouped_mm(self.qdq(selected,mod.fp4_up_global[expert,None]),mod.gate_up_proj[:,mid:,:].transpose(-2,-1),offsets)
            gu=torch.cat((g,u),-1)
        a=mod._apply_gate(gu)
        down=_grouped_mm(self.qdq(a,mod.fp4_down_global[expert,None]),mod.down_proj.transpose(-2,-1),offsets)
        weighted=down*rw.reshape(-1)[perm,None]
        inverse=torch.empty_like(perm);inverse[perm]=torch.arange(perm.numel(),device=perm.device)
        return weighted[inverse].reshape(n,k,-1).sum(1).to(x.dtype)

    def install(self,model,globals_by_module):
        for name,mod in model.named_modules():
            if name.endswith('.mlp.experts'):
                for proj in ['gate','up','down']:
                    values=torch.stack([globals_by_module[f'{name}.{e}.{proj}_proj'] for e in range(mod.num_experts)]).flatten().cuda()
                    mod.register_buffer(f'fp4_{proj}_global',values,persistent=False)
                mod.fp4_gate_up_equal=torch.equal(mod.fp4_gate_global,mod.fp4_up_global)
                def forward(m,x,idx,rw):return self.experts_forward(m,x,idx,rw)
                mod.forward=types.MethodType(forward,mod)
            elif isinstance(mod,torch.nn.Linear) and name in globals_by_module:
                mod.register_buffer('fp4_input_global',globals_by_module[name].cuda(),persistent=False)
                def pre(m,args):return (self.qdq(args[0],m.fp4_input_global),*args[1:])
                mod.register_forward_pre_hook(pre)

def load(device='cuda:0',emit=lambda *a,**k:None):
    config_path=hf_hub_download(MODEL,'config.json',revision=REVISION)
    config=json.load(open(config_path));ref=Reference(config)
    cfg=AutoConfig.from_pretrained(BASE,revision=BASE_REV)
    with init_empty_weights():
        model=AutoModelForImageTextToText.from_config(cfg,dtype=torch.bfloat16,attn_implementation='sdpa',experts_implementation='grouped_mm')
    # Published checkpoint omits the unused vision tower. Text-only forward does
    # not access it. All remaining parameters are required and checked below.
    model.model.visual=torch.nn.Identity()
    filename=hf_hub_download(MODEL,'model.safetensors',revision=REVISION)
    globals_by_module={}
    with safe_open(filename,framework='pt',device='cpu') as source:
        keys=set(source.keys())
        def tensor(key):return source.get_tensor(key).to(device)
        for name,param in list(model.named_parameters()):
            if '.mlp.experts.' in name:
                prefix,proj=name.rsplit('.',1)
                modules=['gate_proj','up_proj'] if proj=='gate_up_proj' else ['down_proj']
                value=torch.empty(param.shape,dtype=param.dtype,device=device)
                for part,p in enumerate(modules):
                    names=[f'{prefix}.{e}.{p}' for e in range(param.shape[0])]
                    packed=torch.stack([tensor(s+'.weight_packed') for s in names])
                    scales=torch.stack([tensor(s+'.weight_scale') for s in names])
                    global_w=torch.stack([tensor(s+'.weight_global_scale') for s in names]).reshape(-1,1,1)
                    outrows=packed.shape[1]
                    global_rows=global_w.expand(-1,outrows,-1).reshape(-1,1)
                    decoded=ref.decode(packed,scales,global_rows).reshape(param.shape[0],outrows,-1)
                    value[:,part*outrows:(part+1)*outrows,:].copy_(decoded)
                    for s in names:globals_by_module[s]=source.get_tensor(s+'.input_global_scale')
                    del packed,scales,global_w,global_rows,decoded
            elif name.endswith('.weight') and name[:-6]+'weight_packed' in keys:
                prefix=name[:-7]
                value=ref.decode(tensor(prefix+'.weight_packed'),tensor(prefix+'.weight_scale'),tensor(prefix+'.weight_global_scale'))
                globals_by_module[prefix]=source.get_tensor(prefix+'.input_global_scale')
            else:
                assert name in keys,('Missing required parameter',name)
                value=tensor(name)
            assert tuple(value.shape)==tuple(param.shape),(name,value.shape,param.shape)
            set_module_tensor_to_device(model,name,device,value=value.to(param.dtype))
            del value
            if name.endswith('.mlp.experts.down_proj'):emit('layer_loaded',name=name,allocated_gb=torch.cuda.memory_allocated()/1e9)
    # Preserve initialized rotary buffers rather than using to_empty().
    for name,buffer in list(model.named_buffers()):
        assert not buffer.is_meta,('Uninitialized buffer',name)
        set_module_tensor_to_device(model,name,device,value=buffer.to(device))
    assert not any(p.is_meta for p in model.parameters())
    ref.install(model,globals_by_module)
    return model.eval(),ref
