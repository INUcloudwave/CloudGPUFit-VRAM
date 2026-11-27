"""Discriminating-condition summary from per_condition.csv (aws_check.py output). usage: discriminating.py <3090 retry csv>"""
import csv, sys
ret = set(r['condition_id'] for r in csv.DictReader(open(sys.argv[1])))
CNN = ('resnet-18', 'resnet-50', 'mobilenet-v2-1.0-224')
rows = [r for r in csv.DictReader(open('per_condition.csv')) if r['retry'] == 'False' and r['same_alloc_as_3090'] == 'True'
        and r['case_id'] not in ret and r['model'] not in CNN]
print("same allocated stream as RTX 3090, no retry on either GPU, non-CNN")
print("gpu | group | n | reserved==3090 | |diff|<=2MiB | discriminating(stock!=desc, 3090==v3 exact): AWS==v3")
for g in sorted({r['gpu'] for r in rows}):
    for grp, f in (('all', lambda r: True), ('t5 (eager attn)', lambda r: r['model'].startswith('t5')),
                   ('sdpa models', lambda r: not r['model'].startswith('t5'))):
        rs = [r for r in rows if r['gpu'] == g and f(r)]
        eq = sum(r['aws_res'] == r['rtx3090_res'] for r in rs)
        le2 = sum(abs(float(r['aws_minus_3090_MiB'])) <= 2 for r in rs)
        disc = [r for r in rs if r['stock_nows'] and r['stock_nows'] != r['desc_nows'] and r['rtx3090_res'] == r['v3_R_inf']]
        dok = sum(r['aws_res'] == r['v3_R_inf'] for r in disc)
        print(f"{g} | {grp} | {len(rs)} | {eq} | {le2} | {dok}/{len(disc)}")
