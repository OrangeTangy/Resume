"""Independent audit of labels, reported selection counts, and run provenance."""
import hashlib,json,re
from pathlib import Path
R=Path(__file__).resolve().parent
for root in [R,R/'replication-1.5b']:
    data=json.loads((root/'data.json').read_text());analysis=json.loads((root/'analysis.json').read_text())
    assert sum(x['split']=='test' for x in data)==80
    assert sum(x['split']=='dev' for x in data)==40
    assert len({x['question'] for x in data})==120
    for x in data:
        ns=list(map(int,re.findall(r'\d+',x['question'])))
        expected=[lambda v:v[0]+v[1],lambda v:v[0]*v[1],lambda v:(v[0]+v[1])*v[2],lambda v:v[0]*v[1]-v[2]][x['family']](ns)
        assert expected==x['answer'] and len(set(x['candidates']))==16
        assert x['candidates'].count(expected)==1
    for mode in ['native','8','4']:
        run=json.loads((root/f'scores-{mode}.json').read_text())
        for name,key in [('data.json','data_sha256'),('protocol.md','protocol_sha256'),('run_experiment.py','script_sha256')]:
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==run[key],(root,name)
        for n in [1,2,4,8,16]:
            correct=sum(x['candidates'][max(range(n),key=lambda i:run['scores'][j][i])]==x['answer'] for j,x in enumerate(data) if x['split']=='test')
            row=next(x for x in analysis['summary'] if x['precision']==mode and x['n']==n)
            assert correct==row['correct'] and row['accuracy']==correct/80
    assert all(x['violations']==0 for x in analysis['ex_post_bound_checks'])
    print('PASS',root.name,': labels, splits, counts, provenance, stability checks')
assert (R/'data.json').read_bytes()==(R/'replication-1.5b/data.json').read_bytes()
print('PASS identical candidate pools across model sizes')
