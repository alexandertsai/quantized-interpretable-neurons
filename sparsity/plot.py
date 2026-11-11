import json,pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=pathlib.Path(__file__).resolve().parent
data={p:json.loads((root/f'results_{p}.json').read_text()) for p in ['bf16','fp8']}
metadata=json.loads((root/'environment.json').read_text())
n=len(data['bf16']['conditions']['unmasked']['window_losses'])
indices=np.random.default_rng(42).integers(0,n,size=(3000,n))
fig,(ax,control)=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={'width_ratios':[1.6,1]})
colors={'bf16':'#245b91','fp8':'#d47823'}
for precision in ['bf16','fp8']:
    d=data[precision]['conditions'];base=np.array(d['unmasked']['window_losses'])
    values=[];lo=[];hi=[]
    for k in [256,320,512,1024]:
        row=d[f'top{k}'];values.append(row['relative_loss_increase_percent'])
        boot=100*(np.array(row['window_losses'])[indices].mean(1)/base[indices].mean(1)-1)
        low,high=np.percentile(boot,[2.5,97.5]);lo.append(low);hi.append(high)
    values=np.array(values)
    ax.errorbar(np.array([256,320,512,1024])/4096*100,values,yerr=[values-np.array(lo),np.array(hi)-values],
        color=colors[precision],marker='o' if precision=='bf16' else 'D',label=precision.upper(),capsize=3,lw=2)
    positions=np.arange(2)+(-0.18 if precision=='bf16' else 0.18)
    heights=[d[k]['delta_loss_nats'] for k in ['top320','random320']]
    bars=control.bar(positions,heights,width=.34,color=colors[precision],label=precision.upper())
    for bar,key in zip(bars,['top320','random320']):
        control.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.06,f"+{d[key]['relative_loss_increase_percent']:.1f}%",ha='center',fontsize=9,color=colors[precision])
ax.set_title('Magnitude-selected activations',loc='left',fontsize=12,fontweight='bold')
ax.set_xlabel('Routed activations retained (%)')
ax.set_ylabel('Increase in next-token loss (%)')
ax.set_xticks(np.array([256,320,512,1024])/4096*100,['6.25','7.81\n(top-320)','12.5','25'])
ax.axvline(320/4096*100,color='#777777',ls=':',lw=1,zorder=0)
ax.legend(frameon=False)
control.set_title('Same budget, different selection',loc='left',fontsize=12,fontweight='bold')
control.set_xticks([0,1],['Top-320','Random-320'])
control.set_ylabel('Additional loss (nats/token)')
control.set_ylim(0,max(data[p]['conditions']['random320']['delta_loss_nats'] for p in data)*1.2)
for axis in [ax,control]:
    axis.spines[['top','right']].set_visible(False);axis.grid(axis='y',alpha=.18);axis.set_axisbelow(True)
fig.suptitle('BF16 vs official FP8: neuron activation sparsity',x=.07,ha='left',fontweight='bold',fontsize=15)
fig.text(.07,.025,f"{n} WikiText-2 test articles · {metadata['sequence_length']} tokens/article · no training · shared experts unchanged\nError bars: 95% bootstrap intervals over articles. Semantic interpretability remains untested.",fontsize=9,color='#444444')
fig.subplots_adjust(left=.07,right=.98,bottom=.24,top=.82,wspace=.38)
fig.savefig(root/'sparsity-comparison.png',dpi=180)
fig.savefig(root/'sparsity-comparison.svg')
