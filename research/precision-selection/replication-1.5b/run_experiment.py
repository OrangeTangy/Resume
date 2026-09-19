"""Local MLX pilot. Run from any cwd with --model /path/to/checkpoint."""
import argparse, hashlib, json, time, platform
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def build_data():
    rng=np.random.default_rng(20260919); rows=[]; seen=set()
    for family in range(4):
        for i in range(30):
            while True:
                a,b,c=map(int,rng.integers(3,40,3))
                if family==0: question=f'What is {a} + {b}?'; answer=a+b
                elif family==1: question=f'What is {a} * {b}?'; answer=a*b
                elif family==2: question=f'What is ({a} + {b}) * {c}?'; answer=(a+b)*c
                else: question=f'What is {a} * {b} - {c}?'; answer=a*b-c
                if question not in seen: seen.add(question); break
            offsets=rng.choice(np.array([x for x in range(-30,31) if x]),15,replace=False)
            candidates=[answer]+[answer+int(x) for x in offsets]; rng.shuffle(candidates)
            rows.append(dict(id=f'f{family}-{i:02d}',family=family,split='dev' if i%3==0 else 'test',question=question,answer=answer,candidates=candidates))
    return rows

def prompt(row,candidate):
    return [{'role':'system','content':'You are an arithmetic verifier. Answer only Yes or No.'},
            {'role':'user','content':f'Question: {row["question"]}\nProposed answer: {candidate}\nIs the proposed answer correct?'}]

def run(args):
    import mlx.core as mx
    import mlx.nn as nn
    mx.set_cache_limit(128 * 1024 * 1024)
    from mlx.utils import tree_flatten
    from mlx_lm import load
    rows=build_data(); data_path=ROOT/'data.json'
    serialized=json.dumps(rows,indent=2)+'\n'
    if data_path.exists(): assert data_path.read_text()==serialized
    else: data_path.write_text(serialized)
    t=time.perf_counter(); model,tok=load(args.model); model.eval()
    dtypes=sorted(set(str(v.dtype) for _,v in tree_flatten(model.parameters())))
    if args.bits: nn.quantize(model,group_size=64,bits=args.bits)
    mx.eval(model.parameters()); load_s=time.perf_counter()-t
    param_bytes=sum(v.nbytes for _,v in tree_flatten(model.parameters()))
    yes=tok.encode('Yes',add_special_tokens=False); no=tok.encode('No',add_special_tokens=False)
    assert len(yes)==len(no)==1, (yes,no)
    seqs=[tok.apply_chat_template(prompt(r,c),tokenize=True,add_generation_prompt=True) for r in rows for c in r['candidates']]
    def score(batch):
        lengths=np.array([len(x) for x in batch]); width=int(max(lengths))
        x=mx.array([s+[tok.pad_token_id]*(width-len(s)) for s in batch])
        h=model.model(x)[mx.arange(len(batch)),mx.array(lengths-1)]
        logits=model.model.embed_tokens.as_linear(h) if model.args.tie_word_embeddings else model.lm_head(h)
        s=logits[:,yes[0]].astype(mx.float32)-logits[:,no[0]].astype(mx.float32)
        mx.eval(s); return np.array(s)
    warm=score(seqs[:4]); solo=np.array([score([s])[0] for s in seqs[:4]])
    batch_delta=float(np.max(abs(warm-solo)))
    # BF16 arithmetic can round differently with matrix shape. Preserve observed difference.
    if batch_delta>0.25: raise RuntimeError(f'Unexpected batch/solo deviation: {batch_delta}')
    out=[]; timings=[]; t=time.perf_counter()
    for i in range(0,len(seqs),4):
        t0=time.perf_counter(); out.extend(score(seqs[i:i+4]).tolist()); timings.append(time.perf_counter()-t0)
        if i%160==0: print(f'{args.bits or "native"}: {i}/{len(seqs)} candidates; {time.perf_counter()-t:.1f}s',flush=True)
    elapsed=time.perf_counter()-t
    result=dict(allocator_cache_limit_bytes=128*1024*1024,bits=args.bits or 'native',group_size=64 if args.bits else None,checkpoint_dtypes=dtypes,
                data_sha256=hashlib.sha256(serialized.encode()).hexdigest(),
                protocol_sha256=hashlib.sha256((ROOT/'protocol.md').read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                model=json.loads((ROOT/'model.json').read_text()),python=platform.python_version(),
                scoring_seconds=elapsed,load_quantize_seconds=load_s,parameter_bytes=param_bytes,
                total_input_tokens=sum(map(len,seqs)),candidate_count=len(seqs),batch_size=4,
                warmup_batch_single_max_abs_difference=batch_delta,label_tokens={'Yes':yes,'No':no},
                batch_seconds=timings,scores=np.array(out).reshape(len(rows),16).tolist())
    name=f'scores-{args.bits or "native"}.json';(ROOT/name).write_text(json.dumps(result,indent=2)+'\n')
    print(name,elapsed,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--bits',type=int,choices=[4,8]);run(p.parse_args())
