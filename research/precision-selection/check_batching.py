"""Post-protocol numerical control: native model, singleton batches, held-out set."""
import json,time
from pathlib import Path
import numpy as np
import mlx.core as mx
from mlx_lm import load
from run_experiment import prompt
R=Path(__file__).resolve().parent
D=json.loads((R/'data.json').read_text()); model,tok=load('work/model');model.eval()
yes=tok.encode('Yes',add_special_tokens=False)[0];no=tok.encode('No',add_special_tokens=False)[0]
scores=[];t=time.perf_counter()
for j,row in enumerate(D):
    if row['split']!='test':continue
    one=[]
    for c in row['candidates']:
        seq=tok.apply_chat_template(prompt(row,c),tokenize=True,add_generation_prompt=True)
        h=model.model(mx.array([seq]))[:,-1,:]
        logits=model.model.embed_tokens.as_linear(h)
        s=logits[:,yes].astype(mx.float32)-logits[:,no].astype(mx.float32);mx.eval(s);one.append(float(s.item()))
    scores.append(one)
    if j%10==1:print(f'singleton control {j}/{len(D)}: {time.perf_counter()-t:.1f}s',flush=True)
mask=np.array([r['split']=='test' for r in D]);base=np.array(json.loads((R/'scores-native.json').read_text())['scores'])[mask];s=np.array(scores)
C=np.array([r['candidates'] for r in D])[mask];truth=np.array([r['answer'] for r in D])[mask]
summary=[]
for n in [1,2,4,8,16]:
    ix=s[:,:n].argmax(1);bi=base[:,:n].argmax(1)
    summary.append(dict(n=n,selection_disagreement=float((ix!=bi).mean()),accuracy=float((C[np.arange(len(C)),ix]==truth).mean())))
(R/'batching-control.json').write_text(json.dumps(dict(scope='Post-protocol native singleton-batch sensitivity control on the held-out questions. Does not replace primary results.',seconds=time.perf_counter()-t,scores=scores,max_score_change=float(abs(s-base).max()),summary=summary),indent=2)+'\n')
print(summary,flush=True)
