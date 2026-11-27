#!/usr/bin/env python3
"""Export existing CPU prediction records; never runs inference or AWS jobs."""
import collections,csv,datetime,hashlib,json,math,shutil,statistics,zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
from zoneinfo import ZoneInfo

BASE=Path('/home/inuos/cloudgpufit-predictions')
CODE=Path('/home/inuos/CloudGPUFit-cpu-singlepass-20261108')
OUT=Path('/home/inuos/CloudGPUFit_CPU_predictions_20261110')
SOURCES=[BASE/'parallel-all342-20261108-180550',BASE/'parallel-all342-20261108-183250']
MIB=1024**2;GIB=1024**3

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def jread(path):return json.loads(path.read_text())
def jwrite(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def copy(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
def div(value,unit):return value/unit if value is not None else None

def csvfile(name,rows,fields):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

FIELDS=[
('case_id','조건 식별자; AWS 실측의 case_id와 대응'),('status','ok 또는 error'),
('workload_type','decoder_llm / text_encdec / vision_cls'),('model','Hugging Face 모델 ID'),('size','small / medium / large'),('dtype','fp16 / fp32 / bf16; 실제 CPU 실행 정밀도'),
('batch_size','한 요청의 배치'),('seq_len','decoder LLM 입력 토큰 수; 다른 유형은 빈칸'),('encoder_seq_len','T5 인코더 입력 길이'),('decoder_seq_len','T5 디코더 prefix 길이'),('max_new_tokens','고정 생성 토큰 수; Vision은 빈칸'),('resolution','Vision 입력 해상도'),('mode','generate 또는 prefill'),
('predicted_allocated_bytes','xMem allocator의 tensor allocated peak; CPU RSS가 아님'),('predicted_reserved_bytes','xMem allocator의 reserved peak; allocated와 별도 비교'),('predicted_allocated_MiB','allocated bytes / 2^20'),('predicted_reserved_MiB','reserved bytes / 2^20'),('predicted_allocated_GiB','allocated bytes / 2^30'),('predicted_reserved_GiB','reserved bytes / 2^30'),
('model_storage_bytes','모델 parameter bytes; GPU allocator 전체 사용량과 다름'),('resident_storage_bytes','모델 parameter/buffer의 고유 상주 storage bytes'),('input_storage_bytes','입력 고유 storage bytes'),
('self_kv_storage_bytes','관측된 self KV underlying storage 최대값'),('cross_kv_storage_bytes','관측된 cross KV underlying storage 최대값'),('self_kv_logical_bytes','self KV numel × element_size 최대값'),('cross_kv_logical_bytes','cross KV logical bytes 최대값'),
('warmup_invocations','성공 결과는 0회'),('measured_invocations','성공 결과는 1회'),('tensor_lifetime_count','수명 매칭된 tensor allocation 수'),('profiler_event_count','Chrome trace 전체 profiler event 수'),
('final_attempt_seconds','최종 시도 wall time; queue 대기 제외'),('previous_attempt_seconds','재시도 전 기록된 첫 시도 시간'),('recorded_attempts_seconds','결과에 기록된 시도 시간의 합계; 병렬 전체 wall time 아님'),('attempt_count','최종 시도 번호; 이전 source에 필드가 없으면 1'),
('max_logged_cpu_rss_GiB','최종 시도 phase 로그에 기록된 최대 CPU RSS; OOM/timeout이면 실제 최종 peak보다 작을 수 있음; VRAM 아님'),('max_logged_cgroup_peak_GiB','최종 시도 phase 로그의 최대 cgroup peak; 미관측 종료 peak보다 작을 수 있음'),('job_ram_limit_GiB','최종 시도 컨테이너 RAM 상한; VRAM 예측값 아님'),
('error_kind','최종 실패 분류; 성공이면 빈칸'),('error_stage','결과에 기록된 마지막 단계'),('error','오류 메시지'),('exit_code','docker CLI 종료 코드'),('oom_killed','Docker OOMKilled 플래그'),
('prediction_version','예측 코드 의미 버전'),('revision','고정된 모델 revision'),('actual_input_shapes_json','실제로 실행한 입력 형태 JSON'),('source_run','원본 실행 폴더명'),('cpu_threads','조건당 CPU 스레드 수'),('source_jobs','해당 batch의 최대 병렬 수'),
('raw_prediction','패키지 내 원본 prediction JSON 상대경로'),('raw_input','패키지 내 요청·config JSON 상대경로'),('final_log','패키지 내 최종 시도 로그 상대경로'),('input_config_sha256','요청 model_config의 정렬된 JSON SHA-256')]
FIELDNAMES=[f[0] for f in FIELDS]

def xlsx(path,sheets):
    ns='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    relns='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    def tag(n):return '{'+ns+'}'+n
    def col(n):
        s=''
        while n:n,a=divmod(n-1,26);s=chr(65+a)+s
        return s
    def xml(el):return ET.tostring(el,encoding='utf-8',xml_declaration=True)
    types=ET.Element('Types',xmlns='http://schemas.openxmlformats.org/package/2006/content-types')
    for ext,typ in [('rels','application/vnd.openxmlformats-package.relationships+xml'),('xml','application/xml')]:ET.SubElement(types,'Default',Extension=ext,ContentType=typ)
    for part,typ in [('workbook.xml','sheet.main+xml'),('styles.xml','styles+xml')]:ET.SubElement(types,'Override',PartName='/xl/'+part,ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.'+typ)
    book=ET.Element(tag('workbook'));views=ET.SubElement(book,tag('bookViews'));ET.SubElement(views,tag('workbookView'),activeTab='0')
    sl=ET.SubElement(book,tag('sheets'))
    rel=ET.Element('Relationships',xmlns='http://schemas.openxmlformats.org/package/2006/relationships')
    mainrel=ET.Element('Relationships',xmlns='http://schemas.openxmlformats.org/package/2006/relationships')
    ET.SubElement(mainrel,'Relationship',Id='rId1',Type=relns+'/officeDocument',Target='xl/workbook.xml')
    style='''<?xml version="1.0" encoding="UTF-8"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Calibri"/></font></fonts><fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FF24476A"/><bgColor indexed="64"/></patternFill></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="3"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyAlignment="1"><alignment wrapText="1" vertical="center"/></xf><xf numFmtId="2" fontId="0" fillId="0" borderId="0" xfId="0" applyNumberFormat="1"/></cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>'''
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('_rels/.rels',xml(mainrel));z.writestr('xl/styles.xml',style)
        for i,(name,headers,rows) in enumerate(sheets,1):
            assert len(name)<=31
            ET.SubElement(sl,tag('sheet'),name=name,sheetId=str(i),attrib={'{'+relns+'}id':'rId'+str(i)})
            ET.SubElement(rel,'Relationship',Id='rId'+str(i),Type=relns+'/worksheet',Target=f'worksheets/sheet{i}.xml')
            ET.SubElement(types,'Override',PartName=f'/xl/worksheets/sheet{i}.xml',ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml')
            sheet=ET.Element(tag('worksheet'));ET.SubElement(sheet,tag('dimension'),ref=f'A1:{col(len(headers))}{len(rows)+1}')
            sv=ET.SubElement(sheet,tag('sheetViews'));view=ET.SubElement(sv,tag('sheetView'),workbookViewId='0',zoomScale='85')
            ET.SubElement(view,tag('pane'),ySplit='1',topLeftCell='A2',activePane='bottomLeft',state='frozen')
            ET.SubElement(sheet,tag('sheetFormatPr'),defaultRowHeight='15')
            cols=ET.SubElement(sheet,tag('cols'))
            for j,h in enumerate(headers,1):
                width=54 if h in ('case_id','조건ID','설명','내용','원본실행') else 34 if 'model' in h or h=='모델' else 20
                if len(headers)<=3:width=35 if j==1 else 105
                ET.SubElement(cols,tag('col'),min=str(j),max=str(j),width=str(width),customWidth='1')
            data=ET.SubElement(sheet,tag('sheetData'))
            for n,values in enumerate([headers]+rows,1):
                row=ET.SubElement(data,tag('row'),r=str(n))
                if n==1:row.set('ht','32');row.set('customHeight','1')
                for j,value in enumerate(values,1):
                    if value is None:continue
                    cell=ET.SubElement(row,tag('c'),r=f'{col(j)}{n}')
                    if n==1:cell.set('s','1')
                    if isinstance(value,bool):cell.set('t','b');ET.SubElement(cell,tag('v')).text='1' if value else '0'
                    elif isinstance(value,(int,float)):
                        assert not isinstance(value,float) or math.isfinite(value)
                        if isinstance(value,float) and n!=1:cell.set('s','2')
                        ET.SubElement(cell,tag('v')).text=str(value)
                    else:
                        cell.set('t','inlineStr');txt=ET.SubElement(ET.SubElement(cell,tag('is')),tag('t'));txt.set('{http://www.w3.org/XML/1998/namespace}space','preserve');txt.text=str(value)
            ET.SubElement(sheet,tag('autoFilter'),ref=f'A1:{col(len(headers))}{len(rows)+1}')
            z.writestr(f'xl/worksheets/sheet{i}.xml',xml(sheet))
        ET.SubElement(rel,'Relationship',Id='rIdStyles',Type=relns+'/styles',Target='styles.xml')
        z.writestr('xl/workbook.xml',xml(book));z.writestr('xl/_rels/workbook.xml.rels',xml(rel));z.writestr('[Content_Types].xml',xml(types))


def main():
    assert not OUT.exists(),'Output folder already exists; preserve existing export'
    OUT.mkdir()
    manifests=[jread(s/'manifest.json') for s in SOURCES]
    assert manifests[0]['identity']==manifests[1]['identity']
    assert manifests[0]['plan_sha256']==manifests[1]['plan_sha256']
    assert manifests[0]['revisions_sha256']==manifests[1]['revisions_sha256']
    plan=jread(CODE/'benchmarks/aws342/selected-plan.json');cases={c['case_id']:c for c in plan['cases'] if c['suite']=='core'}
    assert len(cases)==342
    chosen={}
    for i,src in enumerate(SOURCES):
        for file in src.glob('*.prediction.json'):
            p=jread(file)
            if i==0 and p.get('status')!='ok':continue
            cid=p['case_id'];assert cid not in chosen;chosen[cid]=(src,file,p,i)
    assert set(chosen)==set(cases)
    rows=[];raw_cases=[];provenance=[];gpu_rows=[]
    for cid,c in cases.items():
        src,file,p,i=chosen[cid];inputs=jread(src/(cid+'.input.json'));req=inputs['request'];trace=p.get('trace_summary',{})
        assert req['model']==c['model'] and req['dtype']==c['dtype']
        assert all(req[k]==v for k,v in c['params'].items())
        assert inputs['revision']==p['revision']
        if p['status']=='ok':
            assert p['prediction_version']=='4-native-inference-single-request'
            assert trace['warmup_invocations']==0 and trace['measured_invocations']==1
        rawpred='raw/predictions/'+file.name;rawinput='raw/predictions/'+cid+'.input.json'
        copy(file,OUT/rawpred);copy(src/(cid+'.input.json'),OUT/rawinput)
        logs=list(src.glob(cid+'*.log'))
        for log in logs:copy(log,OUT/'raw/logs'/log.name)
        attempt=p.get('attempt',1);finalname=cid+(f'.attempt{attempt}' if attempt>1 else '')+'.log';finallog=src/finalname
        assert finallog.exists(),finallog
        events=[]
        for line in finallog.read_text(errors='replace').splitlines():
            if line.startswith('CGF_PROGRESS '):events.append(json.loads(line[13:]))
        prev=p.get('previous_attempt',{}).get('runner_elapsed_seconds',0) or 0;elapsed=p.get('runner_elapsed_seconds')
        row={'case_id':cid,'status':p['status'],'workload_type':c['workload_type'],'model':c['model'],'size':c['size'],'dtype':c['dtype'],
            **{k:c['params'].get(k) for k in ['batch_size','seq_len','encoder_seq_len','decoder_seq_len','max_new_tokens','resolution']},'mode':req['mode'],
            **{k:p.get(k) for k in ['predicted_allocated_bytes','predicted_reserved_bytes','model_storage_bytes']},
            'predicted_allocated_MiB':div(p.get('predicted_allocated_bytes'),MIB),'predicted_reserved_MiB':div(p.get('predicted_reserved_bytes'),MIB),
            'predicted_allocated_GiB':div(p.get('predicted_allocated_bytes'),GIB),'predicted_reserved_GiB':div(p.get('predicted_reserved_bytes'),GIB),
            **{k:trace.get(k) for k in ['resident_storage_bytes','input_storage_bytes','self_kv_storage_bytes','cross_kv_storage_bytes','self_kv_logical_bytes','cross_kv_logical_bytes','warmup_invocations','measured_invocations','tensor_lifetime_count']},
            'profiler_event_count':trace.get('total_profiler_events'),'final_attempt_seconds':elapsed,'previous_attempt_seconds':prev if attempt>1 else None,
            'recorded_attempts_seconds':(elapsed+prev) if elapsed is not None else None,'attempt_count':attempt,
            'max_logged_cpu_rss_GiB':div(max((e.get('max_rss_bytes',0) for e in events),default=0),GIB) if events else None,
            'max_logged_cgroup_peak_GiB':div(max((e.get('cgroup_memory_peak_bytes',0) for e in events),default=0),GIB) if events else None,
            'job_ram_limit_GiB':div(p.get('memory_limit_bytes',p.get('container',{}).get('memory_limit_bytes')),GIB),
            'error_kind':p.get('error_kind'),'error_stage':p.get('stage'),'error':p.get('error'),'exit_code':p.get('exit_code'),'oom_killed':p.get('oom_killed'),
            'prediction_version':p.get('prediction_version') or '4-native-inference-single-request (source identity)', 'revision':p['revision'],
            'actual_input_shapes_json':json.dumps(p['actual_input_shapes'],sort_keys=True) if 'actual_input_shapes' in p else None,
            'source_run':src.name,'cpu_threads':req['threads'],'source_jobs':manifests[i]['jobs'],
            'raw_prediction':rawpred,'raw_input':rawinput,'final_log':'raw/logs/'+finalname,
            'input_config_sha256':hashlib.sha256(json.dumps(req['model_config'],sort_keys=True).encode()).hexdigest()}
        assert set(row)==set(FIELDNAMES),(set(row)-set(FIELDNAMES),set(FIELDNAMES)-set(row))
        if p['status']!='ok':assert row['predicted_allocated_bytes'] is None and row['predicted_reserved_bytes'] is None
        rows.append(row);raw_cases.append({'case':c,'request_record':inputs,'prediction':p})
        provenance.append({'case_id':cid,'source_prediction_path':str(file),'source_input_path':str(src/(cid+'.input.json')),'source_run':src.name,
            'prediction_sha256':sha(file),'input_sha256':sha(src/(cid+'.input.json')),'included_prediction':rawpred,'included_input':rawinput,'included_logs':['raw/logs/'+f.name for f in logs]})
        for gpu,g in p.get('gpu_requirements',{}).items():
            gpu_rows.append({'case_id':cid,'gpu':gpu,'dtype':c['dtype'],'status':g['status'],'allocated_bytes':g.get('allocated_bytes'),'reserved_bytes':g.get('reserved_bytes'),
              'outside_estimate_bytes':g.get('outside_estimate_bytes'),'outside_observed_upper_bytes':g.get('outside_observed_upper_bytes'),
              'estimated_nvml_bytes':g.get('estimated_nvml_bytes'),'required_vram_bytes':g.get('required_vram_bytes'),'required_vram_GiB':div(g.get('required_vram_bytes'),GIB),
              'calibration_id':g.get('calibration_id'),'policy':g.get('policy')})
    success=[r for r in rows if r['status']=='ok'];errors=[r for r in rows if r['status']!='ok'];assert len(success)==325 and len(errors)==17
    errorcounts=collections.Counter(r['error_kind'] for r in errors);assert errorcounts=={'cgroup_oom':16,'timeout':1}
    csvfile('cpu_predictions_all.csv',rows,FIELDNAMES);csvfile('cpu_predictions_success.csv',success,FIELDNAMES);csvfile('cpu_predictions_failures.csv',errors,FIELDNAMES)
    jwrite(OUT/'cpu_predictions.json',{'schema_version':1,'scope':'CPU-only prediction records; no AWS actual measurements included','total_cases':342,'success_count':325,'failure_count':17,'rows':rows})
    jwrite(OUT/'raw_predictions_with_requests.json',{'schema_version':1,'cases':raw_cases})
    summary=[]
    for model in dict.fromkeys(c['model'] for c in cases.values()):
        for dtype in ['fp32','fp16','bf16']:
            group=[r for r in rows if r['model']==model and r['dtype']==dtype];ok=[r for r in group if r['status']=='ok'];fail=[r for r in group if r['status']!='ok']
            summary.append({'model':model,'dtype':dtype,'total':len(group),'success':len(ok),'cgroup_oom':sum(r['error_kind']=='cgroup_oom' for r in fail),'timeout':sum(r['error_kind']=='timeout' for r in fail),
             'success_rate_pct':100*len(ok)/len(group),'median_success_seconds':statistics.median(r['final_attempt_seconds'] for r in ok) if ok else None,
             'max_success_seconds':max((r['final_attempt_seconds'] for r in ok),default=None),'max_predicted_allocated_GiB':max((r['predicted_allocated_GiB'] for r in ok),default=None),'max_predicted_reserved_GiB':max((r['predicted_reserved_GiB'] for r in ok),default=None)})
    csvfile('model_dtype_summary.csv',summary,list(summary[0]));csvfile('gpu_requirements.csv',gpu_rows,list(gpu_rows[0]));jwrite(OUT/'prediction_provenance.json',{'sources':[str(s) for s in SOURCES],'cases':provenance})
    for src in SOURCES:
        for name in ['manifest.json','scheduler.jsonl']:
            if (src/name).exists():copy(src/name,OUT/'raw/runs'/src.name/name)
    for name,expected in manifests[1]['identity'].items():
        assert sha(CODE/name)==expected;copy(CODE/name,OUT/'source/predictor'/name)
    for name in ['selected-plan.json','revisions.json','xmem-allocator-lock.json']:copy(CODE/'benchmarks/aws342'/name,OUT/'source'/name)
    assert sha(OUT/'source/selected-plan.json')==manifests[1]['plan_sha256'];assert sha(OUT/'source/revisions.json')==manifests[1]['revisions_sha256']
    runner=BASE/'oom-unlimited-audit-20261110-103935/run_prediction_recovery.before.py'
    assert sha(runner)==manifests[1]['runner_sha256'];copy(runner,OUT/'source/runner_for_332.py')
    assert sha(CODE/'tools/prediction_scheduler.py')==manifests[1]['scheduler_sha256'];copy(CODE/'tools/prediction_scheduler.py',OUT/'source/scheduler_for_332.py')
    copy(BASE/'retry_oom_unlimited_20261110.sh',OUT/'commands/retry_oom_unlimited.sh')
    instructions=[
      ['전체 조건 수',342],['정상 예측',325],['최종 실패',17],['RAM 제한 초과',16],['시간 초과',1],['미실행',0],
      ['예측 버전','4-native-inference-single-request'],['CPU 실행','warmup 0회 / 측정 추론 1회 / native dtype / full generate'],
      ['주요 비교값','predicted_allocated_bytes, predicted_reserved_bytes'],['단위','MiB=2^20 bytes, GiB=2^30 bytes. *_bytes가 원본 정확값입니다.'],
      ['주의','CPU RSS 및 job RAM 상한을 GPU VRAM 예측값으로 사용하지 마세요.'],['실패 값','실패 조건의 예측값은 빈칸/null이며 0으로 대체하지 않았습니다.'],
      ['GPU 요구량','GPU별 표는 기존 outside 보정값을 더한 파생 추정치이며 AWS 실측 결과가 아닙니다. L40S 보정은 포함하지 않습니다.'],
      ['CPU/GPU 차이','CPU는 from_config 초기화 가중치와 합성 입력, GPU 실측은 pretrained 가중치. shape/revision은 확인하되 값까지 같다고 간주하지 마세요.'],
      ['실행 이력 차이','CPU는 단일 요청/빈 allocator+상주 storage. GPU hybrid의 선행 performance/warmup 캐시와 reserved 초기 상태가 다릅니다.'],
      ['출처','앞선 실행 성공 10개 + 후속 실행 332개(성공315/실패17). 동일 predictor/plan/revision hash 확인.'],
      ['재실행','제가 시작한 무제한 RAM 재실행은 중단했고 이 파일에 포함하지 않았습니다. 명령은 commands/retry_oom_unlimited.sh에만 있습니다.']]
    shortkeys=['case_id','workload_type','model','size','dtype','batch_size','seq_len','encoder_seq_len','decoder_seq_len','max_new_tokens','resolution','predicted_allocated_MiB','predicted_reserved_MiB','final_attempt_seconds','attempt_count','max_logged_cpu_rss_GiB']
    shortheaders=['조건ID','유형','모델','규모','정밀도','배치','입력길이','인코더길이','디코더prefix','생성토큰','해상도','allocated예측_MiB','reserved예측_MiB','최종시도_초','시도횟수','로그CPU_RSS_GiB']
    failkeys=['case_id','model','dtype','batch_size','seq_len','encoder_seq_len','decoder_seq_len','max_new_tokens','error_kind','error_stage','job_ram_limit_GiB','final_attempt_seconds','error']
    runrows=[]
    for src,m in zip(SOURCES,manifests):runrows.append([src.name,m['status'],m['total_cases'],m['completed_cases'],m['success_count'],m['jobs'],m['elapsed_seconds'],m['runner_sha256']])
    xlsx(OUT/'CPU_predictions.xlsx',[
      ('요약',['항목','내용'],instructions),('성공325',shortheaders,[[r[k] for k in shortkeys] for r in success]),
      ('전체342',FIELDNAMES,[[r[k] for k in FIELDNAMES] for r in rows]),('실패17',failkeys,[[r[k] for k in failkeys] for r in errors]),
      ('모델정밀도별',list(summary[0]),[list(r.values()) for r in summary]),('GPU별요구량',list(gpu_rows[0]),[list(r.values()) for r in gpu_rows]),
      ('컬럼설명',['컬럼','설명'],[list(x) for x in FIELDS]),('출처',['원본실행','원본상태','대상수','처리수','성공수','최대병렬','실행벽시간_초','runner_sha256'],runrows)])
    readme='''# CloudGPUFit CPU 예측 결과 — 2026-11-10 정리

## 바로 확인할 파일

- `CPU_predictions.xlsx`: 요약, 성공325, 전체342, 실패17, 모델/정밀도별, GPU별 요구량, 컬럼설명, 원본 출처 시트. 첫 행 고정과 자동 필터를 적용했습니다.
- `cpu_predictions_success.csv`: AWS 실측과 우선 비교할 정상 예측 325개.
- `cpu_predictions_all.csv`: 전체 조건 342개. 실패 17개도 보존.
- `cpu_predictions_failures.csv`: cgroup_oom 16개 + timeout 1개.
- `cpu_predictions.json`: CSV와 같은 정리 결과. 실패의 예측값은 null.
- `raw_predictions_with_requests.json`: 조건·요청·모델 config·원본 결과를 한 JSON으로 묶었습니다.
- `raw/predictions/`: 원본 prediction/input JSON 342쌍. 기존 비교 스크립트의 --predictions 입력으로 사용할 수 있습니다.
- `raw/logs/`: 선택된 조건의 최초/solo 재시도 로그. 원본 파일을 수정하지 않았습니다.
- `prediction_provenance.json`, `raw/runs/`: 각 조건의 원본 위치·hash와 두 실행의 manifest/scheduler 기록.
- `source/`: 실제 predictor hash와 일치하는 코드·계획·revision·allocator lock. runner/scheduler는 후속 332개 실행과 hash가 일치하는 버전입니다.
- `commands/retry_oom_unlimited.sh`: 사용자 직접 실행용 명령. 이 패키지를 만들면서 실행하지 않았습니다.

## 집계 범위

앞선 실행 `parallel-all342-20261108-180550`에서 성공한 10개와, 이를 제외한 후속 실행 `parallel-all342-20261108-183250`의 332개 결과를 case_id로 합쳤습니다. 중복/누락은 없습니다. 후속 실행은 성공315/실패17이고, 전체는 **성공325 / 실패17 / 미실행0**입니다.

첫 실행 manifest는 중단으로 runner_failed여도, 이 패키지에는 해당 실행의 실제 정상 결과 10개만 채택했습니다. 후속 실행의 재시도 전 오류는 최종 결과의 previous_attempt 및 원본 로그에 남아 있습니다. 예전 native-v3 결과 244개나 작은 검증용 모델 결과는 섞지 않았습니다. 제가 직접 시작했다가 사용자 요청으로 중단한 11월10일 RAM 무제한 시도도 포함하지 않았습니다.

두 실행의 predictor 5개 파일 identity, selected-plan 및 revisions SHA-256은 동일합니다. 호스트 실행기/병렬·보호 설정은 달랐으며 원본 manifest에서 확인할 수 있습니다. 성공 결과 모두 prediction_version=4-native-inference-single-request, warmup=0, measured=1입니다. manifest의 git_commit은 작업 기준 7f822d7이며, 실제 변경 소스는 identity hash를 기준으로 확인해야 합니다.

## AWS 실측과 비교하는 방법

1. case_id, model, dtype, revision, batch/길이/해상도/max_new_tokens 및 actual_input_shapes를 대응합니다. case_id 번호만 보고 입력을 추정하지 마세요.
2. CPU `predicted_allocated_bytes` ↔ GPU `inference_peak_allocated_bytes`를 비교합니다.
3. CPU `predicted_reserved_bytes` ↔ GPU `inference_peak_reserved_bytes`를 별도로 비교합니다.
4. NVML 비교가 필요하면 기존 GPU별 보정 범위를 확인하고 `reserved + outside` 추정값을 GPU `nvml_process_sampled_peak_bytes`와 비교합니다. allocated를 NVML과 직접 같은 지표처럼 비교하지 마세요.
5. GPU 실측도 정상인 조건만 오차 통계에 사용하고, CPU 실패/실측 미지원/누락은 별도로 집계하세요. T4 BF16 미지원과 L40S 보정 미포함을 주의하세요.

CPU는 임의 초기화 from_config 모델, GPU는 pretrained 모델이며 입력 난수 값까지 동일한 것은 아닙니다. CPU는 단일 요청을 빈 allocator+상주 storage에서 재생하지만, 기존 GPU hybrid memory 측정은 선행 performance/warmup 캐시를 이어받았습니다. reserved 차이를 모두 추론 텐서 오차 또는 모두 캐시 효과라고 단정하면 안 됩니다. 이 파일에는 **AWS 실측이나 CPU-GPU 오차 통계는 포함하지 않았습니다**.

## 단위와 실행시간

- 정확한 비교는 *_bytes 정수로 하세요. MiB=2^20 bytes, GiB=2^30 bytes입니다.
- CPU RSS, cgroup peak, job_ram_limit_GiB는 CPU 실행 자원입니다. GPU VRAM 예측값이 아닙니다.
- max_logged_*는 로그에 남은 수치입니다. OOM/timeout 직전 값은 기록되지 않을 수 있어 실패 조건의 실제 peak라고 해석하지 마세요.
- final_attempt_seconds는 최종 시도 wall time입니다. 재시도 포함 시간은 recorded_attempts_seconds이며 대기시간은 제외됩니다. 병렬 조건 시간을 모두 합쳐 전체 실험 wall time이라고 부르지 마세요.
- GPU별 required_vram은 기존 보정표 기반 파생 추정입니다. 전 GPU에 보장되는 상한이나 새 실측값이 아닙니다.

## 직접 OOM 재실행

현재 VM에 SSH로 접속한 tmux에서 다음 명령을 직접 실행하세요. 이 명령은 기존 최종 OOM 16개만, 한 번에 한 조건씩, RAM hard limit/저메모리 강제종료 보호 없이 실행합니다. 조건당 6시간 제한은 남습니다. 약47GiB 물리 RAM을 초과하면 커널 OOM이 발생할 수 있습니다.

```bash
bash /home/inuos/cloudgpufit-predictions/retry_oom_unlimited_20261110.sh
```

다른 노트북에서는 패키지의 결과 파일을 확인하시면 됩니다. 재실행 스크립트는 준비된 INUOSVM-1 경로·Docker snapshot이 있는 VM용입니다.

## 무결성

`SHA256SUMS.txt`에 결과·원본·코드 파일의 SHA-256을 넣었습니다. 각 원본 prediction/input의 SHA-256은 prediction_provenance.json에도 기록했습니다. 원본 실험 폴더를 덮어쓰지 않았습니다.
'''
    (OUT/'README_먼저읽기.md').write_text(readme)
    copy(Path(__file__),OUT/'source/export_cpu_results.py')
    manifest={'schema_version':1,'created_at':datetime.datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),'total_cases':342,'success_count':325,'error_count':17,'error_kinds':dict(errorcounts),
      'prediction_version':'4-native-inference-single-request','source_runs':[str(s) for s in SOURCES],'predictor_identity':manifests[1]['identity'],'plan_sha256':manifests[1]['plan_sha256'],'revisions_sha256':manifests[1]['revisions_sha256'],
      'cpu_inference_started_for_export':False,'aws_actuals_included':False,'gpu_requirement_rows':len(gpu_rows),'warnings':['Failure predictions are null, never zero.','CPU RAM metrics are not GPU VRAM predictions.','Only 325 successful CPU predictions available; no AWS error statistics included.']}
    jwrite(OUT/'export_manifest.json',manifest)
    checks=[(sha(p),str(p.relative_to(OUT))) for p in sorted(OUT.rglob('*')) if p.is_file()]
    (OUT/'SHA256SUMS.txt').write_text(''.join(f'{digest}  {name}\n' for digest,name in checks))
    package=OUT.with_suffix('.zip');assert not package.exists()
    with zipfile.ZipFile(package,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,str(Path(OUT.name)/p.relative_to(OUT)))
    print('Created',package,'bytes',package.stat().st_size,'sha256',sha(package))
    print('XLSX',OUT/'CPU_predictions.xlsx')
    print('Rows',len(rows),'success',len(success),'errors',dict(errorcounts),'GPU derived rows',len(gpu_rows),'files',len(checks)+1)

if __name__=='__main__':main()
