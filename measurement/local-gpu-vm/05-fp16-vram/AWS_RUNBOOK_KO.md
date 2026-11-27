# CloudGPUFit FP16 VRAM — AWS 이관 런북 (g6 우선)

- 작성: 2026-11-11, INUGPU (RTX 3090)
- 기준 실행 방식: **v2** (FP16, Transformers 기본 attention = `attn_implementation` 인자 생략)

## 0-0. 2026-11-15 변경 (현재 기본값)

- 기본 조건표: **`config/conditions_main395_l256.json`** (sha256 `d4a9b09caff3064c00c04358bf2e462996bced23de93da98652bd1aba04b7289`)
  - OPT·TinyLlama·Qwen, T5 3종의 입력×생성 길이 후보: {128,512,1024}² → **{128,256,512}²**. 1024는 실행 시간이 너무 길어서 뺐습니다.
  - GPT-2, CNN, ViT, 배치, 모델·revision, worker·runner 코드는 바꾸지 않았습니다. 총 395개입니다.
  - 생성기 `config/generate_main_l256.py`
    - 680표에 있는 245개는 원본 객체와 똑같은지 확인합니다.
    - 256이 들어간 150개는 같은 형식으로 새로 만듭니다.
  - 파일럿 26개는 그대로 포함됩니다.
- label: `aws-<타입>__l256`. 이전 1024 실행 결과(`aws-<타입>`)와 섞이지 않습니다.
- 이전 1024 조건표로 돌리려면: `CGF_CONDITIONS=conditions_main395.json bash RUN_ON_AWS.sh <타입>` (label `aws-<타입>`)
- 아래 §0은 이전 조건표에 대한 설명입니다. 규칙은 1024가 256으로 바뀐 것 말고는 같습니다.

## 0. 조건표 (이전: 2026-11-11)

- 파일: `config/conditions_main395.json`
- 조건 수: **395개**
  - 요청 본문은 "355개"였지만, 그룹별로 적힌 개수를 더하면 135+60+135+60+5 = 395입니다.
  - 그룹별 규칙대로 만들었고 개수를 임의로 줄이지 않았습니다.

  | 그룹 | 모델 | 축 | 후보 | 조건 수 |
  |---|---|---|---|---|
  | 디코더 LLM | OPT-125M, TinyLlama, Qwen2.5-1.5B | 입력×생성 | {128,512,1024}² | 135 |
  | GPT-2 | GPT-2, Medium, XL | 입력×생성 | {128,512}² | 60 |
  | 인코더-디코더 | T5-small, Base, 3B | 입력×생성 | {128,512,1024}² | 135 |
  | CNN | ResNet-18/50, MobileNet V2 | 해상도 | 128/224/256/512 | 60 |
  | ViT | ViT Large | 해상도 | 224 고정 | 5 |

  - 모든 그룹의 배치: 1/8/16/32/128
- 680개 표(`config/conditions.json`)의 부분집합입니다. 조건 객체와 condition_id는 원본 그대로 복사했습니다.
- 26개 파일럿(`config/pilot.json`)은 모두 포함되어 있습니다.
- 모델 ID, revision, 입력·생성 규칙(`config/models.json`)은 바꾸지 않았습니다.
  - 주의: `models.json`의 `common.attn_implementation: "eager"` 항목은 680표 생성 당시의 설명 텍스트입니다. worker는 이 값을 쓰지 않고 `--attn` 인자만 따릅니다.

## 0-1. 한 파일 실행 (2026-11-15 갱신, 권장)

- 받은 파일: `cgf-fp16-aws-full-20261115.tar` 1개 (코드·설정·매니페스트 + Docker 이미지 포함)
- 실행:

```bash
tar xf cgf-fp16-aws-full-20261115.tar && cd cgf-fp16-aws-full-20261115
tmux new -s cgf
bash RUN_ON_AWS.sh g6.xlarge        # 두 번째 인자로 캐시 경로 지정 가능 (기본 ./hf-cache)
# 이전 번들로 받은 모델 캐시가 남아 있으면 재사용 (28GB 재다운로드 방지):
# bash RUN_ON_AWS.sh g6.xlarge ~/cgf-fp16-aws-full-20261112/hf-cache
```

- 진행 순서
  1. 체크섬 확인
  2. 인스턴스 타입 확인 (다르면 중단)
  3. 이미지 로드 및 검증 (이미 같은 이미지가 있으면 건너뜀)
  4. 모델 캐시 검증. 실패하면 고정 revision으로 내려받은 뒤 다시 전체 해시 검증
     - 캐시는 용량(28GB) 때문에 파일에 넣지 않았습니다. AWS가 huggingface.co에 접속할 수 있어야 합니다.
  5. 파일럿 26개 실행. PASS하지 못하면 여기서 중단
  6. 본실험 395개 실행 (measure + validate)
- 중단되면 같은 명령을 다시 실행하면 이어서 진행됩니다.
- 본실험에서 validate를 생략하려면 `CGF_MAIN_VALIDATE=0 bash RUN_ON_AWS.sh g6.xlarge`로 실행합니다.
- 로그: `cgf-fp16-vram-20261111/logs/RUN_ON_AWS_aws-<타입>__l256.txt`
- 결과: `cgf-fp16-vram-20261111/runs/aws-<타입>__l256/` (이전 조건표는 `aws-<타입>/`)

## 1. 필요한 것

- GPU 1개 인스턴스 (runner가 GPU 1개를 강제)
- NVIDIA 드라이버, Docker, nvidia-container-toolkit (`--gpus` 동작)
- python3: 호스트 runner용, 표준 라이브러리만 사용
- 디스크 여유 60GB 이상 권장
  - 이미지: 압축 해제 시 약 9.8GB, 로드 중에는 +3.4GB
  - 모델 캐시: 28.0GB (27,989,184,806 B)
  - 결과물

## 2. 순서 (모든 장기 실행은 tmux 안에서)

```bash
EXP=~/cgf/cgf-fp16-vram-20261111        # 번들 tar.gz를 ~/cgf 에 풀었을 때
CACHE=~/cgf/hf-cache
cd $EXP && sha256sum -c MANIFEST_SHA256SUMS

# (1) 배정 인스턴스 확인 — 다르면 중단
bash aws/aws_00_preflight.sh g6.xlarge

# (2) 이미지 로드 + 검증: 스트림 sha256, 이미지 ID, layer diff_id 대조 → IMAGE_OK true 필요
<INUGPU의 docker save 스트림> | bash aws/aws_01_load_image.sh       # 또는: bash aws/aws_01_load_image.sh <파일>

# (3) 모델 캐시 — 둘 중 하나. 둘 다 마지막에 75개 파일 전체 내용 해시를 검증
bash aws/aws_02_cache.sh download $CACHE     # AWS에서 huggingface.co에 고정 revision으로 직접 받기
bash aws/aws_02_cache.sh verify   $CACHE     # INUGPU에서 스트리밍으로 받은 경우

# (4) 파일럿 26개: measure → validate → 판정. PASS면 runs/<label>/PILOT_PASSED 생성
tmux new -s cgf
bash aws/aws_03_pilot.sh aws-g6.xlarge $CACHE

# (5) 본실험 395개: PILOT_PASSED가 없으면 시작하지 않음. 중단 시 같은 명령을 다시 실행하면 이어서 진행
bash aws/aws_04_main.sh aws-g6.xlarge $CACHE
```

- 파일럿 판정(`tools/check_pilot.py`) 통과 조건
  - 26개 모두 measure ok: 5회 측정 모두 ok, 텍스트 모델은 생성 토큰 수 = 요청값
  - 26개 모두 validate ok, 그리고 all_finite
  - `attn_requested=default`
  - worker sha256 = `d06f0f15…` (v2 검증본)
  - 입력 해시가 RTX 3090 v2 파일럿과 동일
  - VRAM 값은 GPU가 달라 비교하지 않고, 참고용으로 나란히 출력합니다.
  - 자가 점검 결과: RTX 3090 v2 결과는 26/26 PASS, v1 eager 결과는 0/26 FAIL (음성 대조)
- 실패 조건은 그대로 보존됩니다.
  - 자동 재시도나 배치·dtype 변경은 하지 않습니다.
  - 재실행하려면 `--retry-status <status>`를 씁니다. 이전 파일은 `attempts/`로 옮겨지고 삭제되지 않습니다.

## 3. 컨테이너 실행 내용 (runner가 조건마다 새로 띄움)

```
docker run --rm --pull=never --network none --pid=host --gpus device=<GPU UUID> --user <uid>:<gid>
  -e HOME=/tmp -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e CUDA_MODULE_LOADING=LAZY
  -e CGF_IMAGE_ID=<id> -e CGF_GPU_LABEL=<label> -e CGF_INSTANCE_TYPE=<IMDS>
  -v $EXP/src:/src:ro -v $EXP/config:/config:ro -v $CACHE:/hf:ro -v $EXP/runs/<label>/<cid>:/out
  --entrypoint python <image id> -I /src/vram_worker.py --models /config/models.json
  --conditions /config/conditions_main395.json --condition <cid> --mode measure|validate --out /out/<mode>.json --attn default
```

- 결과 위치: `runs/<label>/<cid>/{measure,validate}.json`, `progress.jsonl`, `sessions.jsonl`
- `sessions.jsonl`에 남는 정보: 이미지 ID, GPU, 인스턴스 타입, 조건 파일 sha256, 코드 sha256

## 4. 이미지

`aws/IMAGE_INFO.json` 참고.

- 태그: `cloudgpufit-bench:cu124-v1`
- INUGPU의 이미지 ID는 `sha256:457da255…`입니다.
  - INUGPU Docker는 containerd 이미지 저장소를 쓰기 때문에 이 값은 OCI index digest입니다.
  - 클래식 overlay2 저장소에서는 config digest `sha256:e33dd4e4…`가 ID로 표시됩니다. 둘 다 같은 이미지입니다.
  - layer diff_id 7개가 일치해야 합니다.
- `docker save` 스트림: sha256 `ac217e13f8fad18abadcd70f8431d99f63a5d30e70294f48adab38e19100ef17`, 3,358,278,144 B
  - INUGPU에서 2회 연속 실행해 같은 값이 나오는 것을 확인했습니다.

---

## 0-2. Blackwell(g7) 전용: cu128 이미지 (2026-11-17 추가)

- 이유: g7.2xlarge(RTX PRO 4500 Blackwell, compute capability 12.0)에서 기존 이미지 `cu124-v1`(torch 2.6.0+cu124, sm_50~sm_90)로 pilot 26개가 전부 `compat_or_runtime_error`(3.3초)로 실패. 이 torch에는 sm_120 커널이 없음.
- 새 이미지 `cloudgpufit-bench:cu128-v1` (`aws/Dockerfile.cu128`, `aws/IMAGE_INFO_cu128.json`)
  - torch 2.7.1+cu128 (sm_75~sm_120), cuDNN 9.7.1, cuBLAS 12.8, python 3.11.13, numpy 2.2.6
  - transformers 4.52.4 등 pip로 추가한 패키지는 cu124-v1과 같은 버전으로 고정
- 러너·워커·조건·측정 방식은 그대로. 이미지만 다름 → 라벨에 `__cu128`이 붙어 cu124 결과와 섞이지 않음
- 버전 효과 기준선: 같은 RTX 3090에서 cu128로 측정한 `runs/rtx3090__main395_l256__cu128/` 와 비교해, g7 결과의 차이를 "버전 차이"와 "GPU 차이"로 나눠 봄
- 실행 (cu128 번들은 최상위 `IMAGE_VARIANT` 파일에 cu128이 적혀 있어 기본이 cu128)
  ```bash
  tar xf cgf-fp16-aws-cu128-20261117.tar && cd cgf-fp16-aws-cu128-20261117
  tmux new -s cgf
  bash RUN_ON_AWS.sh g7.2xlarge          # label aws-g7.2xlarge__l256__cu128
  ```
- 드라이버: CUDA 12.8 이상 지원 필요(R570+). AMI "Deep Learning OSS Nvidia Driver AMI GPU PyTorch"의 드라이버 595.x 확인됨. AMI의 PyTorch는 쓰지 않음(측정은 컨테이너 안)
