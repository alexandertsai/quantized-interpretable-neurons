"""Reproduce the pre-activation selection from Goodfire's immutable release."""
import pathlib,json,random,urllib.request,gzip,concurrent.futures
ROOT=pathlib.Path(__file__).resolve().parent
BASE='https://static.goodfire.ai/neuron-atlas/2026-11-09-082204/'
def get(path):
    raw=urllib.request.urlopen(BASE+path,timeout=30).read()
    return json.loads(gzip.decompress(raw) if raw[:2]==b'\x1f\x8b' else raw)
def main():
    idx=get('public_atlas_data/index.json')['neurons'];sc=get('subcircuit_modes_index.json')['poles']
    rng=random.Random(20261111);rows=[];used=set();layers=[9,14,19,24,30,36]
    for sign in [1,-1]:
        for i in range(16):
            layer=layers[i%6];side='+' if sign==1 else '-'
            candidates=[n for n in idx if n['L']==layer and n['did'] not in used and n['poles'].get(side,{}).get('pid') in sc and sc[n['poles'][side]['pid']].get('n_modes',0)>0 and not(layer==14 and n['E']==78)]
            n=rng.choice(candidates);used.add(n['did']);rows.append({'pid':n['poles'][side]['pid'],'layer':layer,'expert':n['E'],'neuron':n['N'],'sign':sign})
    def fetch(row):
        path='subcircuit_modes_data/'+row['pid']+'.json';x=get(path)
        modes=[{'label':m['label'],'description':m.get('description','')} for m in x.get('modes',[]) if not m.get('closed') and not m['label'].lower().startswith('unassigned')]
        assert modes
        return {**row,'source_url':BASE+path,'modes':modes,'description':'\n'.join('- '+m['label']+': '+m['description'] for m in modes)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as e:rows=list(e.map(fetch,rows))
    (ROOT/'descriptions.json').write_text(json.dumps({'selection_seed':20261111,'selection':'32 released poles, 16 positive and 16 negative, stratified over layers 9/14/19/24/30/36, sampled without inspecting our activations','poles':rows},indent=2))
if __name__=='__main__':main()
