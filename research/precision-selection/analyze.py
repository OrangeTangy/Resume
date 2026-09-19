"""Deterministic question-level analysis; no model execution required."""
import json, csv, hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
D=json.loads((R/'data.json').read_text()); C=np.array([x['candidates'] for x in D]); truth=np.array([x['answer'] for x in D]);dev=np.array([x['split']=='dev' for x in D]); test=~dev
runs={k:json.loads((R/f'scores-{k}.json').read_text()) for k in ['native','8','4']}
S={k:np.array(v['scores'],float) for k,v in runs.items()}
for k,v in runs.items():
    assert v['data_sha256']==hashlib.sha256((R/'data.json').read_bytes()).hexdigest()
    assert S[k].shape==(120,16) and np.isfinite(S[k]).all()
assert len(set(x['question'] for x in D))==len(D)
assert all(len(set(x['candidates']))==16 and x['candidates'].count(x['answer'])==1 for x in D)
B=np.random.default_rng(82519).integers(0,test.sum(),size=(10000,test.sum()))
def interval(a):return [float(x) for x in np.quantile(np.asarray(a)[B].mean(1),[.025,.975])]
def selections(s,n):
    ix=s[:,:n].argmax(1); correct=C[np.arange(len(D)),ix]==truth
    return ix,correct
summary=[];policies=[];details=[]
for n in [1,2,4,8,16]:
    ri,rc=selections(S['native'],n);oracle=(C[:,:n]==truth[:,None]).any(1)
    for k in S:
        ix,ok=selections(S[k],n); delta=ok[test].astype(float)-rc[test]
        ties=(S[k][:,:n]==S[k][:,:n].max(1,keepdims=True)).sum(1)>1
        row=dict(n=n,precision=k,correct=int(ok[test].sum()),questions=int(test.sum()),accuracy=float(ok[test].mean()),accuracy_ci=interval(ok[test]),
            delta_reference=float(delta.mean()),delta_reference_ci=interval(delta),disagreement=float((ix[test]!=ri[test]).mean()),
            harmful_flips=int((rc[test]&~ok[test]).sum()),beneficial_flips=int((~rc[test]&ok[test]).sum()),
            oracle_coverage=float(oracle[test].mean()),accuracy_given_available=float(ok[test&oracle].mean()) if oracle[test].any() else None,
            top_tie_rate=float(ties[test].mean()))
        summary.append(row)
        for i in np.where(test)[0]:details.append(dict(id=D[i]['id'],family=D[i]['family'],n=n,precision=k,selected=int(C[i,ix[i]]),correct=bool(ok[i]),oracle=bool(oracle[i]),reference_correct=bool(rc[i]),selection_changed=bool(ix[i]!=ri[i])))
        if k=='native' or n==1:continue
        sorted_s=np.sort(S[k][:,:n],axis=1);gap=sorted_s[:,-1]-sorted_s[:,-2]
        for tag,threshold in [('never',-float('inf'))]+[(f'q{q:g}',float(np.quantile(gap[dev],q))) for q in [0,.25,.5,.75,1]]+[('always',float('inf'))]:
            escalate=gap<=threshold; hybrid=np.where(escalate,rc,ok); hix=np.where(escalate,ri,ix); f=float(escalate[test].mean())
            random_expected=(1-f)*ok[test].astype(float)+f*rc[test].astype(float)
            scorecost=(runs[k]['scoring_seconds']+f*runs['native']['scoring_seconds'])/runs['native']['scoring_seconds']
            policies.append(dict(n=n,precision=k,policy=tag,threshold=None if not np.isfinite(threshold) else threshold,
                escalation_fraction=f,reference_candidate_evaluations=int(escalate[test].sum())*n,
                accuracy=float(hybrid[test].mean()),accuracy_ci=interval(hybrid[test]),
                delta_quantized=float((hybrid[test].astype(float)-ok[test]).mean()),
                delta_quantized_ci=interval(hybrid[test].astype(float)-ok[test]),
                disagreement_reference=float((hix[test]!=ri[test]).mean()),random_rescore_expected_accuracy=float(random_expected.mean()),
                delta_random_expected=float((hybrid[test]-random_expected).mean()),
                delta_random_expected_ci=interval(hybrid[test]-random_expected),estimated_scoring_cost_relative_to_native=scorecost))
# Standard bounded-perturbation condition, assessed only as an ex-post diagnostic.
bounds=[]
for k in ['8','4']:
    for n in [2,4,8,16]:
        s=S['native'][:,:n]; q=S[k][:,:n]; gap=np.sort(s,axis=1)[:,-1]-np.sort(s,axis=1)[:,-2]
        eps=abs(q-s).max(1); certified=gap>2*eps;changed=q.argmax(1)!=s.argmax(1)
        assert not np.any(certified&changed)
        bounds.append(dict(precision=k,n=n,certified_fraction=float(certified[test].mean()),violations=int((certified[test]&changed[test]).sum())))
metrics=dict(summary=summary,policies=policies,ex_post_bound_checks=bounds,n_dev=int(dev.sum()),n_test=int(test.sum()),bootstrap_replicates=10000,
    hardware='Apple M1, 8 GB unified memory',scope='One general instruction model; constructed integer answers; no generation or reasoning traces.',
    candidate_label_accuracy={k:float(((s[test]>0)==(C[test]==truth[test,None])).mean()) for k,s in S.items()},
    always_no_label_accuracy=15/16,
    family_n16=[dict(family=f,precision=k,accuracy=float(selections(s,16)[1][test&np.array([x['family']==f for x in D])].mean())) for f in range(4) for k,s in S.items()])
(R/'analysis.json').write_text(json.dumps(metrics,indent=2)+'\n')
with (R/'question-results.csv').open('w') as h:
    w=csv.DictWriter(h,fieldnames=details[0].keys());w.writeheader();w.writerows(details)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors={'native':'#174d78','8':'#139277','4':'#ca6b27'}
fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
for k,label in [('native','Native BF16'),('8','8-bit weights'),('4','4-bit weights')]:
    rows=[r for r in summary if r['precision']==k];xs=[r['n'] for r in rows];ys=np.array([r['accuracy'] for r in rows]);ci=np.array([r['accuracy_ci'] for r in rows])
    axes[0].plot(xs,ys*100,'o-',label=label,color=colors[k]);axes[0].fill_between(xs,ci[:,0]*100,ci[:,1]*100,color=colors[k],alpha=.09)
    if k!='native':axes[1].plot(xs,[100*r['disagreement'] for r in rows],'o-',label=label,color=colors[k])
axes[0].plot(xs,[100*r['oracle_coverage'] for r in rows],'k:',label='Correct answer available')
for ax in axes:ax.set_xscale('log',base=2);ax.set_xticks(xs,labels=xs);ax.set_xlabel('Candidate count N');ax.grid(alpha=.15);ax.legend(fontsize=8)
axes[0].set_ylabel('Selected answer correct (%)');axes[0].set_ylim(0,105);axes[1].set_ylabel('Selection differs from native (%)');axes[1].set_ylim(0,105)
fig.savefig(R/'selection-curves.png',dpi=200);plt.close(fig)
fig,ax=plt.subplots(figsize=(6.8,3.5),layout='constrained')
for k in ['8','4']:
    p=[p for p in policies if p['precision']==k and p['n']==16]
    ax.plot([r['escalation_fraction']*100 for r in p],[r['accuracy']*100 for r in p],'o-',color=colors[k],label=f'{k}-bit + margin gate')
    ax.plot([r['escalation_fraction']*100 for r in p],[r['random_rescore_expected_accuracy']*100 for r in p],'--',color=colors[k],label=f'{k}-bit + random gate (expectation)')
ax.set_xlabel('Held-out questions rescored at native precision (%)');ax.set_ylabel('Selected answer correct (%)');ax.grid(alpha=.15);ax.legend(fontsize=8)
fig.savefig(R/'rescoring.png',dpi=200);plt.close(fig)
print(json.dumps({'n16':[r for r in summary if r['n']==16],'median_n16':[p for p in policies if p['n']==16 and p['policy']=='q0.5']},indent=2))
