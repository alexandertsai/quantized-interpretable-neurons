"""Regenerate the exact public evaluation sample without allocating a GPU."""
import hashlib,json,pathlib,random,re
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT=pathlib.Path(__file__).resolve().parent
def main():
    tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen3.6-35B-A3B',revision='345d467b117d1fd5eda9b879a5988de9ae5f47a7')
    ds=load_dataset('Salesforce/wikitext','wikitext-2-raw-v1',split='test',revision='b08601e04326c79dfdd32d625aee71d232d685c3')
    docs=[];current=[];title='preamble'
    for row in ds:
        line=row['text'];heading=line.strip()
        if re.match(r'^= [^=].*[^=] =$',heading):
            if current:docs.append((title,'\n'.join(current)))
            current=[];title=heading
        current.append(line)
    if current:docs.append((title,'\n'.join(current)))
    rng=random.Random(42);rng.shuffle(docs);ids=[];windows=[]
    for title,text in docs:
        doc_ids=tokenizer(text,add_special_tokens=False)['input_ids']
        if len(doc_ids)<512:continue
        offset=rng.randint(0,len(doc_ids)-512)
        ids.extend(doc_ids[offset:offset+512]);windows.append({'title':title,'offset':offset,'document_tokens':len(doc_ids)})
        if len(windows)==32:break
    assert len(windows)==32
    digest=hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    expected=json.loads((ROOT/'sample_metadata.json').read_text())['sha256']
    assert digest==expected,('Evaluation sample changed',digest,expected)
    data={'ids':ids,'sha256':digest,'document_sampling':True,'windows':windows}
    (ROOT/'tokens_eval.json').write_text(json.dumps(data,indent=2))
    print('Prepared 32 documents / 16,384 input tokens; sample hash matches.')
if __name__=='__main__':main()
