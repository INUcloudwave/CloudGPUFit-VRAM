#!/bin/bash
# INUGPU: build the g4dn env-comparison bundle (one tar: code + config + both images).
#   bash envcmp/make_bundle.sh <reference g4dn L256 runs dir (runs/aws-g4dn.xlarge__l256)> [OUT_PARENT=~/cgf-envcmp-bundle-20261117]
# Steps: select 13x3 conditions from the reference run -> manifest + diag config -> stage files -> MANIFEST_SHA256SUMS
#        -> tar (images streamed from docker save, no loose 4.5 GB copy) -> verify (members, checksums, runner dry run = 39).
set -euo pipefail
EXP=$(readlink -f "$(dirname "$0")/.."); REF=$(readlink -f "${1:?reference runs dir}"); OUTP=${2:-$HOME/cgf-envcmp-bundle-20261117}
NAME=cgf-envcmp-g4dn-20261117; ST=$OUTP/stage/$NAME; E=$ST/cgf-fp16-vram-20261111
rm -rf "$OUTP/stage"; mkdir -p "$E"/{src,src_env,config,config_diag,tools,aws,envcmp,reference}
cd "$EXP"
python3 -I tools/select_envcmp_conditions.py "$REF" config "$E/config/envcmp39_selection"
mv "$E/config/envcmp39_selection_conditions.json" "$E/config/conditions_envcmp39.json"
python3 -I - "$E" <<'PY'
import json, sys
from pathlib import Path
E = Path(sys.argv[1])
m = json.loads((E / "config/conditions_envcmp39.json").read_text())
sel = {r["condition_id"]: r for r in __import__("csv").DictReader((E / "config/envcmp39_selection.csv").open())}
attn_models = ("decoder_llm", "vision_cls")
diag = [c["condition_id"] for c in m["conditions"]
        if (c["workload_type"] == "decoder_llm" or "vit" in c["condition_id"]) and sel[c["condition_id"]]["tier"] in ("light", "heavy")]
(E / "config_diag/conditions.json").write_text(json.dumps({"conditions": m["conditions"], "diag_ids": diag}, indent=1))
print("diag ids", len(diag))
PY
cp config/models.json "$E/config_diag/"
cp src/vram_worker.py src/run_conditions.py "$E/src/"; cp src_env/outside_worker.py "$E/src_env/"
cp config/models.json config/conditions_main395_l256.json config/model_cache_manifest.json config/model_cache_filelist.txt "$E/config/"
cp tools/run_conditions_env.py tools/compare_envs.py tools/select_envcmp_conditions.py tools/diag_attention_kernels.py \
   tools/prepare_cache.py tools/model_cache_filelist.py tools/summarize_runs.py "$E/tools/"
cp aws/aws_00_preflight.sh aws/aws_02_cache.sh aws/IMAGE_INFO.json aws/IMAGES_INFO_envcmp.json aws/Dockerfile.pt251 "$E/aws/"
cp envcmp/host_record.sh envcmp/setup.sh envcmp/run_env.sh envcmp/driver_install_b.sh envcmp/rerun_flagged.sh \
   envcmp/collect.sh envcmp/verify_images.py "$E/envcmp/"
cp envcmp/GUIDE_ENVCMP_KO.md "$E/"
python3 -I - "$E" <<'PY2'
import json, sys
from pathlib import Path
E = Path(sys.argv[1]); sel = json.loads((E / "config/envcmp39_selection.json").read_text())
md = (E / "config/envcmp39_selection.md").read_text(); w = sel["ref_total_wall_s_selected"]
nd = len(json.loads((E / "config_diag/conditions.json").read_text())["diag_ids"])
txt = f"""
## 부록 A. 조건표·예상 시간 (자동 생성, 기준: {sel['reference_run']}, GPU {sel['reference_gpu']}, driver {sel['reference_driver']})

- 선정 규칙: 가벼움 = batch 1 중 최소 peak reserved / 무거움 = 할당기 재시도 0회, peak reserved <= {sel['rules']['heavy_max_frac']:.0%} x 총 메모리, 실행 시간 <= {sel['rules']['wall_cap_s']:.0f}s 중 최대 / 중간 = 같은 필터에서 두 값의 기하평균에 가장 가까운 조건
- 실행 시간 = 기존 실측의 조건당 전체 프로세스 시간(launch wall: 컨테이너 시작 + 모델 로드 + 워밍업 3 + 측정 5)

{md}
| 단계 | 예상 시간 | 근거 |
|---|---|---|
| 환경 1개 측정 (39조건) | {w/60:.0f}분 | 기존 g4dn 실행 시간 합계 {w:.0f}s. 래퍼가 조건당 약 1초(0.2초 대기 3회 + 기록) 추가 |
| 환경 1개 attention 진단 ({nd}조건) | 약 {nd*0.75:.0f}~{nd*1.5:.0f}분 | 추정 (조건당 모델 로드 + 1~2토큰 생성 2회) |
| 호스트 기록 | 약 1분 | 추정 |
| A + C + B 합계 (측정·진단·기록) | 약 {3*(w/60 + nd*1.1 + 1):.0f}분 | 위 합 x 3 |
| 셋업 (이미지 로드, 모델 캐시 27 GB 다운로드·해시 검증) | 약 30~60분 | 추정, 네트워크 속도에 따라 다름 |
| 드라이버 교체 (runfile 다운로드 + DKMS 빌드) + 재부팅 | 약 10~20분 | 추정, 측정 시간과 별도 |
| 재실행 (눈에 띈 조건만) | 조건당 기존 실행 시간 | flagged 수에 따라 다름 |
"""
g = E / "GUIDE_ENVCMP_KO.md"; g.write_text(g.read_text() + txt)
PY2
# reference values of the selected conditions from the previous g4dn run (for A reproducibility, REF= in compare_envs)
RL=$(basename "$REF"); mkdir -p "$E/runs/$RL"
python3 -I -c "import json;print('\n'.join(json.load(open('$E/config/envcmp39_selection.json'))['condition_ids']))" | while read -r c; do
  mkdir -p "$E/runs/$RL/$c"; cp "$REF/$c"/measure.json "$REF/$c"/measure.launch.json "$E/runs/$RL/$c/" 2>/dev/null || true; done
(cd "$E" && find . -type f ! -name MANIFEST_SHA256SUMS -printf '%P\n' | sort | xargs sha256sum > MANIFEST_SHA256SUMS)
cat > "$ST/README_FIRST.txt" <<EOF
g4dn.xlarge driver / PyTorch environment comparison (A -> C -> B), 39 conditions x 3 environments.
Read cgf-fp16-vram-20261111/GUIDE_ENVCMP_KO.md first. Short form:
  bash cgf-fp16-vram-20261111/envcmp/setup.sh ~/hf-cache
  bash cgf-fp16-vram-20261111/envcmp/run_env.sh A ~/hf-cache ; ... run_env.sh C ... ; driver swap + reboot ; ... run_env.sh B ...
  bash cgf-fp16-vram-20261111/envcmp/collect.sh
EOF
(cd "$ST" && sha256sum README_FIRST.txt > SHA256SUMS && echo "e261833b8082fa72700708e633f1592e502412f828ce77bea8fb3ad92809ae79  image/cgf-envcmp-images.docker-save.tar" >> SHA256SUMS)
mkdir -p "$OUTP"; TAR=$OUTP/$NAME.tar
python3 -I - "$OUTP/stage" "$NAME" "$TAR" <<'PY'
import subprocess, sys, tarfile, hashlib, os
stage, name, out = sys.argv[1:4]
SIZE, SHA = 4508747264, "e261833b8082fa72700708e633f1592e502412f828ce77bea8fb3ad92809ae79"
with tarfile.open(out, "w", format=tarfile.GNU_FORMAT) as t:
    t.add(os.path.join(stage, name), arcname=name)
    p = subprocess.Popen(["docker", "save", "cloudgpufit-bench:cu124-v1", "cloudgpufit-bench:pt251-cu124-v1"], stdout=subprocess.PIPE)
    h = hashlib.sha256()
    class R:
        def read(self, n=-1):
            b = p.stdout.read(n); h.update(b); return b
    ti = tarfile.TarInfo(f"{name}/image/cgf-envcmp-images.docker-save.tar"); ti.size = SIZE; ti.mode = 0o644
    t.addfile(ti, R())
    assert p.wait() == 0 and h.hexdigest() == SHA, ("image stream changed", h.hexdigest())
print("tar written", out)
PY
(cd "$OUTP" && sha256sum "$NAME.tar" | tee "$NAME.tar.sha256")
# ---- verify: extract code part to a temp dir, check every checksum, stream-hash the image member, runner dry run
V=$(mktemp -d -p "$OUTP"); tar xf "$TAR" -C "$V" --exclude="*/image/*"
(cd "$V/$NAME" && grep -v image SHA256SUMS | sha256sum -c --quiet && cd cgf-fp16-vram-20261111 && sha256sum -c --quiet MANIFEST_SHA256SUMS) && echo VERIFY_CHECKSUMS_OK
IMG_SHA=$(tar xOf "$TAR" "$NAME/image/cgf-envcmp-images.docker-save.tar" | sha256sum | cut -d' ' -f1)
[ "$IMG_SHA" = e261833b8082fa72700708e633f1592e502412f828ce77bea8fb3ad92809ae79 ] && echo VERIFY_IMAGE_MEMBER_OK
cd "$V/$NAME/cgf-fp16-vram-20261111"
for f in envcmp/*.sh aws/*.sh; do bash -n "$f"; done && echo VERIFY_BASH_SYNTAX_OK
python3 -m py_compile tools/*.py src/*.py src_env/*.py envcmp/*.py && echo VERIFY_PY_COMPILE_OK
python3 -I -c "
import json,collections;m=json.load(open('config/conditions_envcmp39.json'))['conditions'];c=collections.Counter(x['model'] for x in m)
print('models',len(c),'per_model',sorted(set(c.values())),'conditions',len(m),'unique',len({x['condition_id'] for x in m}))
s=json.load(open('config/conditions_main395_l256.json'))['conditions'];S={x['condition_id']:x for x in s}
print('identical_to_main395_l256', all(S[x['condition_id']]==x for x in m))"
python3 -I tools/run_conditions_env.py --conditions conditions_envcmp39.json --list all --mode measure --gpu-label dryrun \
  --attn default --image cloudgpufit-bench:cu124-v1 --env-name A --dry-run --hf-cache /tmp | tail -1
rm -rf "$V" "$OUTP/stage"
ls -l "$TAR"; echo BUNDLE_DONE
