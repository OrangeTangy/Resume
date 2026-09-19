"""Build the paper from measured JSON results; never invent missing results."""
import json, html
from pathlib import Path
import re, subprocess, os
from html.parser import HTMLParser
R=Path(__file__).resolve().parent
A=json.loads((R/'analysis.json').read_text());runs={k:json.loads((R/f'scores-{k}.json').read_text()) for k in ['native','8','4']};control=json.loads((R/'batching-control.json').read_text())
RA=json.loads((R/'replication-1.5b/analysis.json').read_text());RR={k:json.loads((R/f'replication-1.5b/scores-{k}.json').read_text()) for k in ['native','8','4']}
TITLE='Weight Precision and Answer Selection: A Local Pilot with Small Language-Model Judges'
URL='https://tanayanand.me/research/precision-selection/'

tex=[];web=[]
def esc(t):
    return ''.join({'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}.get(c,c) for c in t)
class LatexText(HTMLParser):
    def __init__(self):super().__init__();self.out=[]
    def handle_data(self,t):self.out.append(re.sub(r'\[(\d)\]',lambda m:r'\cite{ref'+m[1]+'}',esc(t)))
    def handle_starttag(self,t,a):
        a=dict(a)
        if t=='b':self.out.append(r'\textbf{')
        elif t=='br':self.out.append(r'\\ ')
        elif t=='link':self.out.append(r'\href{'+a['href']+'}{')
    def handle_endtag(self,t):
        if t in ['b','link']:self.out.append('}')
def conv(t):
    o=LatexText();o.feed(t);return ''.join(o.out)
def p(text,kind='BodyPaper'):
    web.append(f'<h1>{text}</h1>' if kind=='PaperTitle' else f'<p>{text}</p>')
    if kind=='PaperTitle' or text.startswith('WORKING PAPER') or text.startswith('Tanay Anand<br'):return
    if text==abstract:
        tex.append(r'\begin{abstract}'+conv(text)+r'\end{abstract}');return
    if text.startswith('Let s_i'):
        tex.append(r'Let $s_i$ and $q_i$ denote native and quantized scores. For native winner $a$, define \begin{equation}\epsilon=\max_i|q_i-s_i|,\qquad m=s_a-\max_{j\ne a}s_j.\end{equation} If $m>2\epsilon$, the quantized winner is also $a$.');return
    if text.startswith('<b>Proof.'):
        tex.append(r'\paragraph{Proof.} For every competitor $j\ne a$, \begin{equation}q_a-q_j\ge(s_a-\epsilon)-(s_j+\epsilon)\ge m-2\epsilon>0.\end{equation} Thus $a$ remains the unique winner. This standard perturbation argument is included for interpretation, not as a novel theorem.\hfill\rule{0.5em}{0.5em}');return
    tex.append(re.sub(r'([a-f0-9]{40,64})',lambda m:r'\nolinkurl{'+m[1]+'}',conv(text))+'\n')
def h(text):
    web.append(f'<h2>{text}</h2>')
    if text in ['Abstract','References']:return
    level='subsection' if re.match(r'^\d+\.\d+',text) else 'section'
    title=re.sub(r'^\d+(?:\.\d+)?\.?\s*','',text)
    if text.startswith('Appendix:'):tex.append(r'\appendix');title='Audit identifiers'
    tex.append('\\'+level+('' if re.match(r'^\d|Appendix:',text) else '*')+'{'+esc(title)+'}')
def page():pass
def pct(x):return f'{100*x:.1f}'
def ci(x):return f'[{pct(x[0])}, {pct(x[1])}]'
def table(rows,widths):
    spec='@{}'+''.join(r'>{\raggedright\arraybackslash}p{'+str(round(w/sum(widths)*.91,4))+r'\linewidth}' for w in widths)+'@{}'
    content=[]
    for i,row in enumerate(rows):
        content.append(' & '.join(conv(str(c)) for c in row)+r' \\'+(r' \midrule' if i==0 else ''))
    tex.append(r'\begin{center}\small\setlength{\tabcolsep}{3pt}\begin{tabular}{'+spec+'}\n'+r'\toprule'+'\n'+'\n'.join(content)+r'\bottomrule\end{tabular}\end{center}')
    web.append('<div class="tablewrap"><table>'+''.join('<tr>'+''.join(f'<{"th" if i==0 else "td"}>{c}</{"th" if i==0 else "td"}>' for c in row)+'</tr>' for i,row in enumerate(rows))+'</table></div>')
def fig(name,width,height,caption):
    cap=re.sub(r'^Figure \d+\. ','',caption)
    tex.append(r'\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{'+name+r'}\caption{'+conv(cap)+r'}\end{figure}')
    web.append(f'<img src="{name}" alt="{html.escape(caption)}"><p>{caption}</p>')
def row(k,n=16):return next(x for x in A['summary'] if x['precision']==k and x['n']==n)
nat=row('native');q8=row('8');q4=row('4');med={k:next(x for x in A['policies'] if x['precision']==k and x['n']==16 and x['policy']=='q0.5') for k in ['8','4']}
abstract=(f'Weight quantization reduces parameter storage, but its effect on answer selection need not be summarized by candidate-level classification accuracy. '
'We report a controlled local pilot using Qwen2.5-0.5B-Instruct as a prompted arithmetic judge at native BF16, 8-bit and 4-bit weight precision. '
'The experiment evaluates 120 constructed questions, with 40 used for threshold development and 80 held out; each contains 16 distinct integer answers with exactly one correct answer. '
f'At N=16, selected-answer accuracy is {pct(nat["accuracy"])}% at native precision, {pct(q8["accuracy"])}% at 8 bits, and {pct(q4["accuracy"])}% at 4 bits. '
f'The 4-bit versus native paired difference is {pct(q4["delta_reference"])} percentage points, with a 95% question-bootstrap interval of {ci(q4["delta_reference_ci"])}. '
f'We additionally evaluate development-calibrated selective rescoring. Native 0.5B batching changes N=16 selections on {100*control["summary"][-1]["selection_disagreement"]:.1f}% of test questions, revealing substantial implementation sensitivity. '
f'A subsequent 1.5B same-family extension on the same questions yields N=16 accuracies of {100*next(x["accuracy"] for x in RA["summary"] if x["n"]==16 and x["precision"]=="native"):.1f}% (native), {100*next(x["accuracy"] for x in RA["summary"] if x["n"]==16 and x["precision"]=="8"):.1f}% (8-bit), and {100*next(x["accuracy"] for x in RA["summary"] if x["n"]==16 and x["precision"]=="4"):.1f}% (4-bit). This study reports actual quantized-model inference over artificial answer pools, not generated reasoning traces. It establishes neither a general scaling law nor a novel state-of-the-art method.')
p(TITLE,'PaperTitle')
p('Tanay Anand<br/>tanayanand.me','BodyPaper')
h('Abstract');p(abstract)
h('1. Research question and scope')
p('For a fixed candidate pool, how do weight precision and pool size change a judge\'s selected answer? Can a simple score-gap rule selectively invoke the native model to recover its choices? These questions separate three quantities: numerical agreement, answer correctness, and computation cost. None implies either of the others.')
p('This pilot was designed for an Apple M1 laptop with 8 GB of unified memory. Its contribution is an auditable experiment and a reproducible baseline, not a claim of first-ever discovery. The stronger original hypothesis - that generating more answers becomes counterproductive because of judge compression - requires model-generated pools and larger trained verifiers, which are outside the present evidence.')
page()
h('2. Relationship to existing work')
p('Verifier-based selection over math solutions predates this study [1]. Reward-model overoptimization already establishes that stronger optimization of a proxy can hurt the underlying objective [2]. Generative verifiers [3] and recent verification-dynamics experiments [4] motivate evaluating the judge as a separate component. Quasar quantizes the verifier in speculative token decoding [5]; its verification target differs from completed-answer correctness here. These precedents rule out claiming that verifier selection, harmful overoptimization, or quantized verification are themselves new.')
h('3. Experimental design')
p('The plan in protocol.md was written before model scoring. The exact checkpoint is Qwen/Qwen2.5-0.5B-Instruct [6], revision '+runs['native']['model']['revision']+'. Native parameters use BF16. MLX-LM 0.30.2 and MLX 0.29.3 [8] perform affine groupwise weight quantization at 8 or 4 bits with group size 64. Quantizable embeddings and linear layers are included; normalization parameters remain at native precision. The model is not fine-tuned. This is real parameter quantization, not rounding the final scores.')
p('Dataset seed 20260919 generates 30 questions for each of four templates: a+b, a*b, (a+b)*c, and a*b-c, with operands from 3 through 39. Repeated question strings are rejected. Every third question within each template enters the development split, yielding 40 development and 80 test questions. Each pool contains the exact integer answer and 15 unique nonzero offsets sampled without replacement from -30 through 30, then shuffled once. Candidate pools are fixed across precisions. Prefixes of lengths 1, 2, 4, 8 and 16 form the selection tasks.')
p('<b>Important construction effect.</b> At N=16 every question has exactly one correct candidate; smaller prefixes may contain none. Thus oracle coverage grows by construction. These pools are not independent draws from a generator, and the experiment cannot infer whether more LLM generation helps or hurts in deployment. Incorrect answers are simple numerical distractors, not naturally occurring reasoning failures.')
h('3.1 Scoring and deterministic selection')
p('System prompt: "You are an arithmetic verifier. Answer only Yes or No." The user prompt contains the question, a proposed integer answer, and "Is the proposed answer correct?" We apply the official chat template and score the first assistant token as logit(Yes) - logit(No). Token IDs are 9454 and 2753. Both labels are single tokens. This is equivalent to the log odds after restricting the distribution to those labels; it is not a calibrated probability of correctness.')
p('Right-padded batches contain four candidates. Causal attention prevents future padding from influencing real tokens. The final real-token hidden state is gathered before applying the output head. Logits are converted to float32 before subtraction, but this does not undo earlier BF16 rounding. Selection uses the largest score, with ties broken by first position in the pre-shuffled pool. No reasoning tokens are generated.')
h('3.2 Estimands and uncertainty')
p('The unit of evaluation is a held-out question. Correctness comes from integer arithmetic, never from the native model. We report selected-answer accuracy, selection disagreement, beneficial and harmful correctness flips, and answer availability. All intervals use 10,000 paired question-level percentile bootstrap resamples (seed 82519). Candidate-level samples and nested N values are not treated as independent observations. Intervals are descriptive, pointwise and unadjusted for multiple comparisons.')
p('Native, 8-bit and 4-bit runs each score 1,920 candidates (102,576 input tokens). Wall times include the scoring loop after warmup; model loading and quantization are recorded separately. Parameter bytes count model arrays, not process peak memory. Conditions run separately on a shared laptop, so the timing observations do not support deployment-speedup claims.')
page()
h('4. Held-out selection results')
fig('selection-curves.png',492,182,'Figure 1. Left: selected-answer accuracy with pointwise 95% question-bootstrap bands; the dotted curve is answer availability. Right: disagreement with the native model. Constructed pools; 80 held-out questions.')
rows=[['N','Available (%)','Native (%)','8-bit (%)','4-bit (%)','4-bit - native (pp), 95% CI']]
for n in [1,2,4,8,16]:
    rr=row('4',n);rows.append([n,pct(rr['oracle_coverage']),pct(row('native',n)['accuracy']),pct(row('8',n)['accuracy']),pct(rr['accuracy']),f'{pct(rr["delta_reference"])} {ci(rr["delta_reference_ci"])}'])
table(rows,[26,72,64,57,57,216])
p(f'At N=16, the 8-bit model changes the selected candidate on {pct(q8["disagreement"])}% of held-out questions, while 4-bit changes it on {pct(q4["disagreement"])}%. There are {q8["harmful_flips"]} harmful and {q8["beneficial_flips"]} beneficial correctness flips for 8-bit, and {q4["harmful_flips"]} harmful and {q4["beneficial_flips"]} beneficial flips for 4-bit. A changed choice can leave correctness unchanged when both answers are wrong.')
p(f'The native top score is tied on {pct(nat["top_tie_rate"])}% of N=16 test pools; the 8-bit and 4-bit rates are {pct(q8["top_tie_rate"])}% and {pct(q4["top_tie_rate"])}%. The fixed tie-breaking rule is therefore part of the experiment. Quantization need not preserve score scale or break ties in the same way.')
p('These results should not be described as proof that quantization universally harms reasoning or that more candidates cause accuracy collapse. The native judge is fallible, the sample is small, and the candidate construction changes answer availability with N. The full table is retained regardless of whether it supports the motivating hypothesis.')
h('4.1 Candidate classification is an insufficient summary')
p('For comparison, thresholding the score at zero gives candidate-level Yes/No accuracy of '+', '.join(f'{pct(A["candidate_label_accuracy"][k])}% ({k})' for k in ['native','8','4'])+'. However, an always-No classifier obtains 93.75% because 15 of 16 candidates are incorrect. This severe class imbalance makes raw binary accuracy an unsuitable standalone quality measure. Selecting the correct answer is the primary outcome.')
page()
h('5. Selective native rescoring')
p('For each N and quantized precision, compute the gap between its highest and second-highest candidate scores. If that gap is at or below a threshold, replace the quantized selection with the native selection after rescoring all N candidates. Thresholds are the 0th, 25th, 50th, 75th and 100th percentiles of development gaps, plus never and always rescore. Test labels are not used to choose thresholds. The development-median policy is the prespecified primary policy; all thresholds are included in analysis.json.')
fig('rescoring.png',440,226,'Figure 2. N=16 held-out accuracy across all development-calibrated gates. Dashed curves show the expected accuracy of random rescoring at the same test escalation fraction. These are cached-score policy evaluations, not measured online hybrid execution.')
table([['Initial judge','Escalated (%)','Hybrid accuracy (%)','Vs. quantized (pp), 95% CI','Random gate expectation (%)']]+[[k+'-bit',pct(med[k]['escalation_fraction']),pct(med[k]['accuracy']),f'{pct(med[k]["delta_quantized"])} {ci(med[k]["delta_quantized_ci"])}',pct(med[k]['random_rescore_expected_accuracy'])] for k in ['8','4']],[67,75,94,158,98])
p('Random rescoring is compared analytically: at escalation fraction f, its expected per-question correctness is (1-f) times quantized correctness plus f times native correctness. This avoids selecting a favorable random seed. The comparison controls rescoring counts at a fixed N, not exact token-dependent latency. Its paired uncertainty, conditional on the observed escalation fraction, is included in the raw analysis.')
p('The selective policy is a baseline, not an established improvement. Rescoring can remove beneficial quantization changes as well as harmful ones. Agreement with the native model measures numerical fidelity; it does not establish mathematical correctness. A deployed system would also need model residency or loading, scheduling, and end-to-end timing measurements. No energy or production speedup is claimed.')
page()
h('6. A standard stability observation')
p('Let s_i be the native score of candidate i and q_i its quantized score. Suppose every candidate satisfies |q_i - s_i| &lt;= epsilon. If the native winner a has margin m = s_a - max_{j != a} s_j greater than 2 epsilon, the quantized winner is also a.')
p('<b>Proof.</b> For any competitor j, q_a - q_j &gt;= (s_a - epsilon) - (s_j + epsilon) &gt;= m - 2 epsilon &gt; 0. Hence every competitor has a strictly smaller quantized score. Strict inequality excludes ties. This elementary perturbation argument is included for interpretation, not asserted as a novel theorem.')
p('The condition needs a bound on score errors, not just the number of weight bits. It cannot be used as a free inference-time certificate without estimating or certifying those errors. In the present data, epsilon is computed after both precision passes and is only an ex-post diagnostic. The implementation asserts that no pool satisfying the condition has a different winner.')
table([['N','8-bit certified pools (%)','4-bit certified pools (%)']]+[[n,pct(next(x['certified_fraction'] for x in A['ex_post_bound_checks'] if x['precision']=='8' and x['n']==n)),pct(next(x['certified_fraction'] for x in A['ex_post_bound_checks'] if x['precision']=='4' and x['n']==n))] for n in [2,4,8,16]],[52,220,220])
h('6.1 Numerical control added after the primary protocol')
p(f'Initial warmup comparisons between batches and standalone examples differed by up to {runs["native"]["warmup_batch_single_max_abs_difference"]:.3f} score units at native precision. We therefore added a post-protocol control that rescored all 1,280 held-out candidates with the native model in singleton batches. The largest score change was {control["max_score_change"]:.3f}. This is a computational robustness check on the same questions, not an independent replication.')
table([['N','Native batch-1 accuracy (%)','Batch-1 vs. batch-4 selection disagreement (%)']]+[[x['n'],pct(x['accuracy']),pct(x['selection_disagreement'])] for x in control['summary']],[35,180,277])
p(f'At N=16, the native batch-size control changes {pct(control["summary"][-1]["selection_disagreement"])}% of selections, compared with {pct(q8["disagreement"])}% for 8-bit quantization in the primary comparison. Numerical implementation effects are therefore material in this setup.')
p('Batching changes floating-point reduction paths and may affect near ties. The primary precision comparisons keep the input pools, prompts and batch size fixed, but they still combine weight approximation with precision-specific kernel arithmetic. Results are therefore specific to the measured implementation, and singleton-batch results do not replace the primary results after the fact.')
page()
h('6.2 Exploratory extension to a 1.5B judge')
p('After observing the low native accuracy of the 0.5B pilot, we extended the identical scoring protocol to Qwen2.5-1.5B-Instruct [7]. This is an exploratory same-family model-size extension, not an independent confirmation: it reuses the same development and test questions, prompt, candidate ordering and analysis. Both models and all precision settings are reported. No prompts or thresholds were tuned on the larger model test results.')
p('Exact 1.5B checkpoint revision: '+RR['native']['model']['revision']+'. The extension protocol and complete scores are in replication-1.5b/. It uses native BF16, MLX 8-bit and MLX 4-bit weights, with the same group size and batch size. An initial native attempt stopped making logged progress after 640 candidates and was terminated without a completed result. All extension runs use a 128 MiB MLX temporary-buffer cache limit; the execution amendment and aborted log are retained.')
def rep_row(k,n): return next(x for x in RA['summary'] if x['precision']==k and x['n']==n)
table([['N','Native (%)','8-bit (%)','4-bit (%)','4-bit - native (pp), 95% CI']]+[[n,pct(rep_row('native',n)['accuracy']),pct(rep_row('8',n)['accuracy']),pct(rep_row('4',n)['accuracy']),pct(rep_row('4',n)['delta_reference'])+' '+ci(rep_row('4',n)['delta_reference_ci'])] for n in [1,2,4,8,16]],[32,73,68,68,251])
fig('replication-1.5b/selection-curves.png',492,182,'Figure 3. Same-family 1.5B extension, on the same 80 held-out questions. Candidate construction and numerical backend limits remain unchanged.')
r4=rep_row('4',16);r8=rep_row('8',16)
p(f'At N=16, 8-bit selection disagreement is {pct(r8["disagreement"])}%, with {r8["harmful_flips"]} harmful and {r8["beneficial_flips"]} beneficial correctness flips. At 4 bits, disagreement is {pct(r4["disagreement"])}%, with {r4["harmful_flips"]} harmful and {r4["beneficial_flips"]} beneficial flips. These counts separate changes in choice from changes in answer correctness. Equal observed accuracies do not establish equivalence; bootstrap intervals can have zero width when every observed paired correctness difference is zero.')
pm=next(x for x in RA['policies'] if x['n']==16 and x['precision']=='4' and x['policy']=='q0.5')
p(f'The 1.5B 4-bit development-median gate rescores {pct(pm["escalation_fraction"])}% of test pools and reaches {pct(pm["accuracy"])}% accuracy; its paired change versus 4-bit alone is {pct(pm["delta_quantized"])} pp, 95% interval {ci(pm["delta_quantized_ci"])}. All other policy thresholds and the random-gate comparisons are retained in the extension analysis.')
page()
h('7. Resource observations and reproducibility')
table([['Condition','Parameter arrays (MiB)','Scoring (s)','Load + quantize (s)']]+[[k, f'{v["parameter_bytes"]/2**20:.1f}',f'{v["scoring_seconds"]:.2f}',f'{v["load_quantize_seconds"]:.2f}'] for k,v in runs.items()],[82,160,120,130])
table([['1.5B condition','Parameter arrays (MiB)','Scoring (s)','Load + quantize (s)']]+[[k, f'{v["parameter_bytes"]/2**20:.1f}',f'{v["scoring_seconds"]:.2f}',f'{v["load_quantize_seconds"]:.2f}'] for k,v in RR.items()],[82,160,120,130])
p('Measurements are one pass per condition on an Apple M1 with 8 GB unified memory. They are not repeated controlled hardware benchmarks. Activation memory, allocator caches, CPU memory and checkpoint storage are excluded from the parameter-array totals. Model weights are downloaded from the original repository rather than redistributed.')
p('The artifact includes all constructed questions, split labels, candidate scores, per-batch times, complete policy sweeps, question-level outcomes, model revision, environment package lock, source code, and hashes. Running analyze.py regenerates statistics and figures without downloading a model. run_experiment.py performs the native and quantized inference passes. The native singleton control is in check_batching.py.')
p('Artifact and manuscript: <link href="'+URL+'" color="#175986">'+URL+'</link>','SmallPaper')
h('8. Limitations and next study')
p('The primary pilot uses one 0.5B general instruction model, followed by a 1.5B same-family extension on the same questions. It covers four elementary arithmetic templates, one prompt, one candidate-order seed, one quantization backend and 80 test questions. It is neither a standardized reasoning benchmark nor an evaluation of trained reward models. Constructed offsets do not represent realistic generator errors. Bootstrap intervals omit uncertainty across model families, seeds, templates and hardware. BF16 ties and numerical kernel effects complicate attribution. The score-gap gate is heuristic and has no calibrated correctness guarantee.')
p('A stronger follow-up should use multiple trained verifier families and independently generated reasoning traces, replicate across pool seeds, report both low-precision and native correctness under fixed generation budgets, and measure actual hybrid-system latency and memory. Prompt and numeric-kernel controls should be prespecified. The primary hypothesis and sample-size target should be registered before inspecting new test outcomes. If no reliable effect appears, that negative result should remain visible.')
page()
h('9. Conclusion')
p('Weight precision can be studied separately from answer generation by scoring identical candidate pools. This artifact supplies a transparent small-model test, a margin-rescoring baseline and a numerical control. It does not establish that quantization makes additional generation counterproductive, that a new method beats existing research, or that the findings generalize. Broader validation across verifier families and independently generated candidate pools remains necessary.')
h('Acknowledgments')
p('The manuscript received private review from colleagues. Codex assisted with research scoping, literature retrieval, implementation, local experiment execution, analysis, writing, and manuscript preparation.','SmallPaper')
h('References')
refs=[
('Karl Cobbe et al. (2021). Training Verifiers to Solve Math Word Problems. arXiv:2110.14168.','https://arxiv.org/abs/2110.14168'),
('Leo Gao, John Schulman and Jacob Hilton (2022). Scaling Laws for Reward Model Overoptimization. arXiv:2210.10760.','https://arxiv.org/abs/2210.10760'),
('Lunjun Zhang, Arian Hosseini, Hritik Bansal, Mehran Kazemi, Aviral Kumar and Rishabh Agarwal (2025). Generative Verifiers: Reward Modeling as Next-Token Prediction. ICLR. arXiv:2408.15240.','https://arxiv.org/abs/2408.15240'),
('Yefan Zhou, Austin Xu, Yilun Zhou, Janvijay Singh, Jiang Gui and Shafiq Joty (2026). Variation in Verification: Understanding Verification Dynamics in Large Language Models. ICLR. arXiv:2509.17995.','https://arxiv.org/abs/2509.17995'),
('Guang Huang and Zeyi Wen (2026). Quasar: Quantized Self-Speculative Acceleration for Rapid Inference via Memory-Efficient Verification. Preprint, arXiv:2603.01399.','https://arxiv.org/abs/2603.01399'),
('Qwen team. Qwen2.5-0.5B-Instruct model card and checkpoint. Accessed 19 September 2026.','https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct'),
('Qwen team. Qwen2.5-1.5B-Instruct model card and checkpoint. Exploratory extension.','https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct'),
('MLX contributors. MLX-LM: language-model inference on Apple silicon. Runtime used: MLX-LM 0.30.2 / MLX 0.29.3.','https://github.com/ml-explore/mlx-lm')]


entries=[
('Cobbe, Karl and others','Training Verifiers to Solve Math Word Problems','2021','arXiv:2110.14168'),
('Gao, Leo and Schulman, John and Hilton, Jacob','Scaling Laws for Reward Model Overoptimization','2022','arXiv:2210.10760'),
('Zhang, Lunjun and Hosseini, Arian and Bansal, Hritik and Kazemi, Mehran and Kumar, Aviral and Agarwal, Rishabh','Generative Verifiers: Reward Modeling as Next-Token Prediction','2025','ICLR; arXiv:2408.15240'),
('Zhou, Yefan and Xu, Austin and Zhou, Yilun and Singh, Janvijay and Gui, Jiang and Joty, Shafiq','Variation in Verification: Understanding Verification Dynamics in Large Language Models','2026','ICLR; arXiv:2509.17995'),
('Huang, Guang and Wen, Zeyi','Quasar: Quantized Self-Speculative Acceleration for Rapid Inference via Memory-Efficient Verification','2026','Preprint, arXiv:2603.01399'),
('{Qwen team}','Qwen2.5-0.5B-Instruct: Model Card and Checkpoint','2026','Accessed September 19, 2026'),
('{Qwen team}','Qwen2.5-1.5B-Instruct: Model Card and Checkpoint','2026','Accessed September 19, 2026'),
('{MLX contributors}','MLX-LM: Language-Model Inference on Apple Silicon','2026','MLX-LM 0.30.2; MLX 0.29.3')]
bib=[]
for i,((author,title,year,note),(desc,url)) in enumerate(zip(entries,refs),1):
    bib.append('@misc{ref'+str(i)+',\n author = {'+author+'},\n title = {{'+title+'}},\n year = {'+year+'},\n note = {'+note+'},\n howpublished = {\\url{'+url+'}}\n}')
    web.append(f'<p>[{i}] {desc} <a href="{url}">{url}</a></p>')
(R/'references.bib').write_text('\n\n'.join(bib)+'\n')
tex.append(r'\bibliographystyle{unsrt}\bibliography{references}')
h('Appendix: audit identifiers')
p('Dataset SHA-256: '+runs['native']['data_sha256'],'SmallPaper')
p('Protocol SHA-256: '+runs['native']['protocol_sha256'],'SmallPaper')
p('Scoring script SHA-256: '+runs['native']['script_sha256'],'SmallPaper')
p('Exact package versions and source scripts are included with the downloadable research artifact. SHA-256 hashes identify the files used in the primary scoring runs. The subsequent batching control is separately labeled.','SmallPaper')
preamble='\\documentclass[11pt,letterpaper]{article}\n\\usepackage{fontspec}\n\\IfFontExistsTF{Times New Roman}{\\setmainfont{Times New Roman}}{\\setmainfont{TeX Gyre Termes}}\n\\usepackage{unicode-math}\n\\setmathfont{texgyretermes-math.otf}\n\\usepackage[margin=1in]{geometry}\n\\usepackage{amsmath,graphicx,booktabs,array,microtype}\n\\usepackage{xurl}\n\\usepackage[hidelinks]{hyperref}\n\\hypersetup{pdftitle={Weight Precision and Answer Selection: A Local Pilot with Small Language-Model Judges},pdfauthor={Tanay Anand}}\n\\setlength{\\emergencystretch}{2em}\n\\title{Weight Precision and Answer Selection:\\\\A Local Pilot with Small Language-Model Judges}\n\\author{Tanay Anand\\\\\\small\\url{https://tanayanand.me}}\n\\date{19 September 2026}\n\\begin{document}\n\\maketitle\n'
(R/'paper.tex').write_text(preamble+'\n\n'.join(tex)+'\n\\end{document}\n')
subprocess.run([os.environ.get('TECTONIC','tectonic'),'--keep-logs','paper.tex'],cwd=R,check=True)
# HTML is an accessible full-text rendition of the same generated manuscript.
body='\n'.join(web).replace('<link href=','<a href=').replace('</link>','</a>')
(R/'paper.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+TITLE+'</title><style>body{max-width:850px;margin:40px auto;padding:0 24px;font:17px/1.65 "Times New Roman",Times,serif;color:#222}h2{font:700 22px/1.3 "Times New Roman",Times,serif;margin-top:36px}p{overflow-wrap:anywhere}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font:14px/1.5 system-ui}td,th{padding:10px;border-bottom:1px solid #d7e0e8;text-align:left}th{background:#f5f5f5}.tablewrap{overflow-x:auto}a{color:#145a91}</style></head><body><nav><a href="./">Research overview</a> / <a href="paper.pdf">Download PDF</a></nav><main>'+body+'</main></body></html>')
(R/'abstract.txt').write_text(abstract+'\n')
print('Built LaTeX manuscript and full text')
