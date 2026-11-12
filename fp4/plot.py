"""Export a compact precision comparison from the saved numerical results."""
import json,pathlib,argparse
import numpy as np
import matplotlib.pyplot as plt
ROOT=pathlib.Path(__file__).resolve().parent
def main(fp8path=None):
    a=json.loads((ROOT/'analysis.json').read_text());repo=ROOT.parent
    fp8path=pathlib.Path(fp8path) if fp8path else repo/'sparsity'/'results_fp8.json'
    fp8=json.loads(fp8path.read_text())['conditions'];rng=np.random.default_rng(42)
    loss=np.array(fp8['top320']['window_losses']);base=np.array(fp8['unmasked']['window_losses']);idx=rng.integers(0,len(loss),(10000,len(loss)))
    fp8ratio=100*(loss.mean()/base.mean()-1);fp8ci=np.quantile(100*(loss[idx].mean(1)/base[idx].mean(1)-1),[.025,.975])
    entries=[('BF16',a['conditions']['bf16']['adaptive320']['relative_loss_increase_percent']),('FP8',{'estimate':fp8ratio,'document_bootstrap_95ci':fp8ci}),('FP4 weights\nBF16 activations',a['conditions']['fp4_w4a16']['adaptive320']['relative_loss_increase_percent']),('FP4 weights\nFP4 activations',a['conditions']['fp4_w4a4']['adaptive320']['relative_loss_increase_percent'])]
    vals=np.array([e[1]['estimate'] for e in entries]);cis=np.array([e[1]['document_bootstrap_95ci'] for e in entries]);errors=np.stack([vals-cis[:,0],cis[:,1]-vals])
    fig,ax=plt.subplots(figsize=(8,4.7));bars=ax.bar(range(4),vals,yerr=errors,capsize=5,color=['#4a6485','#319a9c','#bc8a4f','#d26b4b'])
    ax.set_xticks(range(4),[e[0] for e in entries]);ax.set_ylabel('Added prediction loss from top-320 filtering (%)');ax.set_title('Neuron sparsity under more aggressive quantization',loc='left',weight='bold',pad=18)
    ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2)
    for bar,v,ci in zip(bars,vals,cis):ax.text(bar.get_x()+bar.get_width()/2,ci[1]+.12,f'{v:.2f}%',ha='center',va='bottom')
    ax.set_ylim(0,max(cis[:,1])+.8)
    fig.text(.11,.02,'32 matched WikiText articles · 95% document bootstrap intervals\nFP4 uses the Red Hat release with QDQ and BF16 matrix multiplication; native kernels untested.',fontsize=9,color='#444444')
    fig.tight_layout(rect=(0,.09,1,1));fig.savefig(ROOT/'precision-comparison.png',dpi=180);fig.savefig(ROOT/'precision-comparison.svg')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fp8-results');main(p.parse_args().fp8_results)
