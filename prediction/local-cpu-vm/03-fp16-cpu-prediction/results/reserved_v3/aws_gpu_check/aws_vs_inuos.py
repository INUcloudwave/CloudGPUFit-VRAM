"""AWS CPU FP16-real stream vs INUOS FP32-shadow stream, both v3 R_inf, vs each GPU's measurement (no-retry conditions on
that GPU). usage: aws_vs_inuos.py <AWS predictions.jsonl>"""
import csv, json, sys, collections
aws = {}
for l in open(sys.argv[1]):
    r = json.loads(l); aws[r['case_id']] = r['main']['R_inf']
CNN = ('resnet-18', 'resnet-50', 'mobilenet-v2-1.0-224')
st = collections.Counter()
for r in csv.DictReader(open('per_condition.csv')):
    if r['retry'] == 'True' or r['case_id'] not in aws: continue
    g = 'cnn' if r['model'] in CNN else ('t5' if r['model'].startswith('t5') else 'llm_vit')
    gpu = r['gpu'].split('-l256')[0]
    m, i, a = int(r['aws_res']), int(r['v3_R_inf']), aws[r['case_id']]
    for grp in (g, 'all_noncnn' if g != 'cnn' else None):
        if not grp: continue
        st[(gpu, grp, 'n')] += 1
        st[(gpu, grp, 'inuos')] += abs(i - m) <= 1e8
        st[(gpu, grp, 'aws')] += abs(a - m) <= 1e8
        st[(gpu, grp, 'same')] += a == i
print("gpu | group | n | INUOS<=100MB | AWS<=100MB | identical predictions")
for gpu in sorted({k[0] for k in st}):
    for grp in ('llm_vit', 't5', 'all_noncnn', 'cnn'):
        print(f"{gpu} | {grp} | {st[(gpu,grp,'n')]} | {st[(gpu,grp,'inuos')]} | {st[(gpu,grp,'aws')]} | {st[(gpu,grp,'same')]}")
