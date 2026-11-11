import pathlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=pathlib.Path(__file__).resolve().parent
a=json.loads((root/'analysis.json').read_text());m=a['mask_transfer'];d=a['description_detection']
fig,(ax,bx)=plt.subplots(1,2,figsize=(11,4.7))
values=np.array([m['transfer_minus_matched_native_loss_nats'],m['transfer_minus_matched_native_kl_nats']])
intervals=np.array([m['loss_difference_95pct_paired_document_bootstrap'],m['kl_difference_95pct_paired_document_bootstrap']])
ax.axhline(0,color='#777777',ls='--',lw=1)
ax.errorbar([0,1],values,yerr=[values-intervals[:,0],intervals[:,1]-values],fmt='o',color='#245b91',capsize=5,ms=7,lw=2)
ax.set_xticks([0,1],['Next-token loss','KL divergence'])
ax.set_ylabel('Transfer minus FP8-selected (nats/token)')
ax.set_xlim(-.5,1.5)
ax.set_title('Causal mask transfer: matched budget',loc='left',fontsize=11,fontweight='bold')
ax.text(.02,.97,f"Mean retained: {m['mean_retained_neurons']:.1f} of 320 neurons\nFP8 clean routing held fixed",transform=ax.transAxes,va='top',fontsize=9,color='#444444')
colors={'bf16':'#245b91','fp8':'#d47823'}
for precision in ['bf16','fp8']:
    x=np.array([0,1])+(-.07 if precision=='bf16' else .07)
    rows=[d['common_pole_metrics'][precision+'_'+kind] for kind in ['released','shuffled_description']]
    v=np.array([r['macro_accuracy'] for r in rows])*100
    c=np.array([r['macro_95pct_pole_bootstrap'] for r in rows])*100
    bx.errorbar(x,v,yerr=[v-c[:,0],c[:,1]-v],fmt='o' if precision=='bf16' else 'D',capsize=4,lw=2,color=colors[precision],label=precision.upper())
bx.axhline(50,color='#777777',ls='--',lw=1,label='Chance')
bx.set_xticks([0,1],['Released descriptions','Shuffled descriptions'])
bx.set_ylabel('Macro detection accuracy (%)')
bx.set_ylim(0,100);bx.set_xlim(-.5,1.5)
bx.set_title('Blind Luna judge: 29 common poles',loc='left',fontsize=11,fontweight='bold')
bx.legend(frameon=False,loc='lower left',fontsize=9)
for axis in [ax,bx]:
    axis.spines[['top','right']].set_visible(False);axis.grid(axis='y',alpha=.15)
fig.suptitle('BF16 → FP8: causal transfer and description prediction',x=.08,ha='left',fontsize=14,fontweight='bold')
fig.text(.08,.025,'No training. Error bars: 95% bootstrap intervals over documents (left) and poles (right).\nEnglish Wikipedia pilot; released-description detection remains inconclusive.',fontsize=9,color='#444444')
fig.subplots_adjust(left=.08,right=.98,top=.82,bottom=.22,wspace=.35)
fig.savefig(root/'followup-comparison.png',dpi=180)
fig.savefig(root/'followup-comparison.svg')
