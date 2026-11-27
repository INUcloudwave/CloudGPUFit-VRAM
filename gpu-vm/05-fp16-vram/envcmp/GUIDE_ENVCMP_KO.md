# g4dn.xlarge 드라이버 · PyTorch 배포 환경 비교 실행 가이드 (A → C → B)

- 목적: 기존 g4dn.xlarge(T4) L256 FP16 실측을 기준으로, **호스트 드라이버**와 **PyTorch 배포 환경**이 VRAM 측정값(allocated, reserved, outside)에 주는 영향을 각각 분리해서 확인한다.
- 범위: 39개 조건(13모델 × 가벼움·중간·무거움) × 환경 3개 = 117케이스. 한 인스턴스에서 A → C → B 순서로 실행한다.
- 조건표·선정 이유·예상 시간: 이 문서 끝의 부록과 `config/envcmp39_selection.md`

## 0. 환경 정의

| 환경 | 호스트 드라이버 | 컨테이너 이미지 | 라벨 (`runs/`) |
|---|---|---|---|
| A 기준 | AMI 기본 드라이버 (이전 실측 595.112.07) | `cloudgpufit-bench:cu124-v1` (torch 2.6.0+cu124, 이전 L256과 같은 이미지 ID) | `aws-g4dn.xlarge__envcmp_A` |
| C PyTorch 비교 | A와 같음 (교체 없음) | `cloudgpufit-bench:pt251-cu124-v1` (torch 2.5.1+cu124) | `aws-g4dn.xlarge__envcmp_C` |
| B 드라이버 비교 | R580 데이터센터 드라이버 (runfile) | A와 같은 이미지 ID | `aws-g4dn.xlarge__envcmp_B` |

- C 이미지는 A 이미지 위에 pip 레이어 하나만 더한 것이다. 바뀐 것은 아래 4개뿐이다.
  - torch 2.6.0 → 2.5.1, torchvision 0.21.0 → 0.20.1, torchaudio 2.6.0 → 2.5.1, triton 3.2.0 → 3.1.0
  - 그대로인 것: python 3.11.11, cuDNN 9.1.0.70, cuBLAS 12.4.5.8 등 nvidia-* 휠 전부, transformers 4.52.4
  - 출처: `aws/IMAGES_INFO_envcmp.json`, 이미지 안 `/opt/pt251_pip_changes.txt`
  - 그래도 torch 빌드 자체(SDPA 커널 선택, 할당기 코드)가 다르므로 C 결과는 "PyTorch 배포 환경 변경"으로 해석한다.
- 공통으로 고정되는 것
  - 측정 코드: `src/vram_worker.py` (sha `d06f0f15`, 이전 L256과 동일) → 워밍업 3 + 측정 5, FP16, seed·입력 동일(입력 sha256 기록)
  - attention: `--attn default` (transformers 기본값)
  - `CUDA_MODULE_LOADING=LAZY`, `PYTORCH_CUDA_ALLOC_CONF` 미설정, torch 스레드 = min(8, vCPU) = 4
  - 모델 revision과 캐시: `config/models.json`, `config/model_cache_manifest.json`. 해시 검증을 통과해야 실행
- 같은 인스턴스, 같은 OS·커널을 쓴다. 커널이 A와 다르면 C·B 실행 스크립트가 중단된다.
- 추가 기록 (래퍼 `src_env/outside_worker.py`, 워커는 수정하지 않음)
  - 세 시점에서 GPU 동기화 직후 다음을 바이트 단위로 기록한다.
    - 시점: CUDA 초기화 직후(`cuda_init`), 모델 로딩 직후(`after_load`), 워밍업 3회 직후(`after_warmup` = 측정 1회차 시작, peak 리셋 전)
    - 기록 항목: PyTorch allocated, reserved, NVML 프로세스 사용량, outside(= NVML − reserved), 할당기 재시도 수
  - 결과 파일은 `runs/<라벨>/<condition_id>/measure.outside.json`이다. 기존 `measure.json`은 그대로 나온다(첫 측정 peak, NVML 피크, OOM 상태 등).
- attention 실제 경로는 두 가지로 구분해서 기록된다.
  - 로그에서 확인한 사실: `measure.json`의 `model.attn_implementation`(transformers가 정한 구현)과 `sdpa_backends_enabled`(켜진 백엔드)
  - 커널 수준: 환경마다 측정이 끝난 뒤 별도 프로세스로 profiler 커널 이름을 수집한다(`runs/<라벨>/_diag/attention_kernels.json`). 측정값과는 무관한 진단이다.
  - T4(sm75)에서는 FlashAttention 커널을 쓸 수 없으므로 메모리 효율 커널이나 math 경로가 예상된다. 이건 추정이고, 커널 로그로 확인해야 한다.

## 1. 인스턴스 준비

- 유형: `g4dn.xlarge` (T4 1개, vCPU 4, RAM 16 GiB)
- AMI: 이전 g4dn L256 실측과 같은 AMI ("Deep Learning OSS Nvidia Driver AMI GPU PyTorch …")
  - 가능하면 같은 AMI ID를 쓴다. AMI ID는 `envcmp/records/*/imds.txt`에 기록된다.
- 루트 볼륨: **100 GB 이상**
  - 모델 캐시 27 GB + 번들 4.6 GB + 로드된 이미지 약 14 GB + 드라이버 runfile
- 인터넷 필요: 모델 다운로드(huggingface.co), B 드라이버 다운로드(us.download.nvidia.com)

## 2. 전송·압축 해제

```bash
# INUGPU에서 (접속 정보만 채우기)
scp -i <키.pem> ~/cgf-envcmp-bundle-20261117/cgf-envcmp-g4dn-20261117.tar ubuntu@<AWS_IP>:~
# AWS에서
cd ~ && sha256sum cgf-envcmp-g4dn-20261117.tar      # 값은 INUGPU의 .tar.sha256과 같아야 함
tar xf cgf-envcmp-g4dn-20261117.tar && cd cgf-envcmp-g4dn-20261117
tmux new -s env
```

## 3. 호스트 사전 확인 (A 실행 전, 1회)

```bash
cd ~/cgf-envcmp-g4dn-20261117
. /etc/os-release; echo "$ID $VERSION_ID"; uname -r
nvidia-smi --query-gpu=name,driver_version,persistence_mode --format=csv
cat /proc/driver/nvidia/version
L=$(ldconfig -p | awk '/libcuda.so.1 /{print $NF; exit}'); readlink -f $L; dpkg -S $(readlink -f $L) 2>/dev/null || rpm -qf $(readlink -f $L) 2>/dev/null || echo "패키지 소유 아님 → runfile 설치"
ls -l /usr/bin/nvidia-uninstall /var/log/nvidia-installer.log 2>&1
V=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
curl -s -o /dev/null -I -w "datacenter runfile: %{http_code}\n" https://us.download.nvidia.com/tesla/$V/NVIDIA-Linux-x86_64-$V.run
curl -s -o /dev/null -I -w "unix driver runfile: %{http_code}\n" https://us.download.nvidia.com/XFree86/Linux-x86_64/$V/NVIDIA-Linux-x86_64-$V.run
# 실험 중 커널·드라이버가 자동으로 바뀌지 않게 (Ubuntu)
sudo systemctl stop unattended-upgrades apt-daily.timer apt-daily-upgrade.timer 2>/dev/null; sudo systemctl disable unattended-upgrades apt-daily.timer apt-daily-upgrade.timer 2>/dev/null
```

- 분기 기준 (A 결과 해석과 B 설치 방법을 정함)

| 확인 결과 | 판정 | B 설치 방법 |
|---|---|---|
| Ubuntu, libcuda가 패키지 소유 아님 (runfile) | 자동 분기 지원 | §5-B 스크립트 (runfile, 커널 모듈 종류 A와 동일) |
| Ubuntu, libcuda가 deb 패키지 소유 | 같은 채널(apt)로 설치해야 함 | §5-B 패키지 분기 |
| Amazon Linux 2023 / rpm | 미지원 | 출력 전체를 공유하고 중단 (명령 별도 제공) |

| 595.112.07 데이터센터 runfile | A의 해석 |
|---|---|
| HTTP 200 | A = 데이터센터 계열 595.112.07로 본다. A↔B = 같은 계열 안에서 595 → 580 버전 차이 |
| 404 (unix driver만 200) | 595.112.07은 데이터센터 배포본이 아니다. A↔B = "AMI 기본 드라이버(595.112.07) → 데이터센터 R580"으로 해석하고 버전 차이만이라고 부르지 않는다. 순수 버전 비교가 필요하면 부록 B의 A′(데이터센터 R595 최신판) 추가 실행 |

## 4. 셋업 (1회)

```bash
cd ~/cgf-envcmp-g4dn-20261117
bash cgf-fp16-vram-20261111/envcmp/setup.sh ~/hf-cache
```

- 체크섬 → 인스턴스 유형 확인(g4dn.xlarge, GPU 1개) → 이미지 2개 로드·ID 검증 → 모델 캐시 다운로드(고정 revision)·해시 검증 → 러너 dry run
- 마지막 줄이 `DRY_RUN selected=39 unique=39`여야 한다.

## 5. 실행

### 5-A. A 기준

```bash
cd ~/cgf-envcmp-g4dn-20261117/cgf-fp16-vram-20261111
bash envcmp/run_env.sh A ~/hf-cache
cat envcmp/records/A_before_*/DRIVER_FACTS.txt
```

### 5-C. PyTorch 비교 (드라이버 그대로)

```bash
bash envcmp/run_env.sh C ~/hf-cache
# A-C 비교와, 눈에 띈 조건만 새 프로세스로 다시 실행 (드라이버를 바꾸기 전에 해야 함)
python3 -I tools/compare_envs.py . config/conditions_envcmp39.json results/envcmp_AC A=runs/aws-g4dn.xlarge__envcmp_A C=runs/aws-g4dn.xlarge__envcmp_C REF=runs/aws-g4dn.xlarge__l256
bash envcmp/rerun_flagged.sh A results/envcmp_AC/flagged_C_vs_A.json 1 ~/hf-cache
bash envcmp/rerun_flagged.sh C results/envcmp_AC/flagged_C_vs_A.json 1 ~/hf-cache
```

- `REF=`는 이전 g4dn L256 원본(번들에 39개만 포함)이다. A가 이전 실측을 재현하는지 함께 본다(`flagged_REF_vs_A.json`).

### 5-B. 드라이버 교체 → 재부팅 → B

```bash
# (1) 설치할 R580 버전 확인: https://docs.nvidia.com/datacenter/tesla/ 의 R580 최신 (2026-11-17 기준 580.196.07)
B_VER=580.196.07
curl -s -o /dev/null -I -w "%{http_code}\n" https://us.download.nvidia.com/tesla/$B_VER/NVIDIA-Linux-x86_64-$B_VER.run   # 200이어야 함
# (2) 커널 헤더 (현재 커널 그대로, 업그레이드 없음)
ls -d /lib/modules/$(uname -r)/build || sudo apt-get install -y linux-headers-$(uname -r)
# (3) runfile 분기: 미리보기 → 적용
bash envcmp/driver_install_b.sh $B_VER
bash envcmp/driver_install_b.sh $B_VER --apply
sudo reboot
```

- 재접속 후 적용 버전과 GPU 접근을 확인한다.

```bash
cd ~/cgf-envcmp-g4dn-20261117/cgf-fp16-vram-20261111
uname -r                                   # A와 같아야 함 (envcmp/records/A_before_*/DRIVER_FACTS.txt)
nvidia-smi --query-gpu=name,driver_version,persistence_mode --format=csv
cat /proc/driver/nvidia/version            # 580.x, 커널 모듈 종류(Open/proprietary)가 A와 같아야 함
docker run --rm --gpus all --entrypoint python cloudgpufit-bench:cu124-v1 -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# persistence mode가 A와 다르면 맞춘다: sudo nvidia-smi -pm 1  (A가 Enabled일 때) / -pm 0
tmux new -s env
bash envcmp/run_env.sh B ~/hf-cache
python3 -I tools/compare_envs.py . config/conditions_envcmp39.json results/envcmp_AB A=runs/aws-g4dn.xlarge__envcmp_A B=runs/aws-g4dn.xlarge__envcmp_B
bash envcmp/rerun_flagged.sh B results/envcmp_AB/flagged_B_vs_A.json 1 ~/hf-cache
```

- docker에서 GPU가 안 잡히면(`False` 또는 오류) CDI 설정을 다시 만든다: `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml && sudo systemctl restart docker`
- runfile 설치가 "kernel module in use"로 실패하면 재부팅한 뒤 바로 (3)의 `--apply`를 다시 실행한다.
- 패키지 분기 (A가 deb 패키지로 설치된 경우, 스크립트는 후보만 출력하고 멈춤)

```bash
apt-cache policy $(dpkg-query -W -f '${Package}\n' 'nvidia-driver-*' 'nvidia-open*' 'cuda-drivers*' 2>/dev/null)   # A 패키지의 저장소 확인
# NVIDIA CUDA 저장소(developer.download.nvidia.com)였다면: 같은 저장소의 R580, A와 같은 모듈 종류(-open 여부)
apt-cache madison nvidia-driver-580-open cuda-drivers-580 nvidia-open-580
sudo apt-get install -y --allow-downgrades nvidia-driver-580-open=<위에서 고른 버전>
# Ubuntu 저장소의 -server 패키지였다면: sudo apt-get install -y nvidia-driver-580-server-open   (A가 proprietary면 -open 없이)
sudo reboot
```

- 이 경우 B는 runfile이 아니라 패키지 버전이다. `envcmp/records/B_before_*/pkgs_deb.txt`에 실제 버전이 남는다.

## 6. 집계·회수

```bash
cd ~/cgf-envcmp-g4dn-20261117/cgf-fp16-vram-20261111
bash envcmp/collect.sh            # 마지막에 RESULT_TAR=~/cgf-envcmp-results-<host>-<시각>.tar.gz
```

- 산출물: `results/envcmp/`
  - `envcmp_long.csv`: 환경×조건 1행
  - `envcmp_wide.csv`: 조건별 B−A, C−A 차이 (바이트)
  - `envcmp_summary.md`: 상태 변화, 차이 통계, 눈에 띈 조건
  - `flagged_*_vs_A.json`
- tar에 들어가는 것: 모든 envcmp 라벨 원본(재실행 포함), `envcmp/records`(호스트·드라이버 기록), `logs`

```bash
# 노트북 또는 INUGPU에서 회수
scp -i <키.pem> 'ubuntu@<AWS_IP>:~/cgf-envcmp-results-*.tar.gz*' .
sha256sum -c cgf-envcmp-results-*.tar.gz.sha256
```

## 7. 해석 규칙

- 같은 프로세스 안의 측정 5회는 독립 실행 5회가 아니다. 조건마다 독립 표본은 1개다(재실행하면 2개).
- 같은 환경에서도 reserved가 두 값으로 갈리는 현상이 있다(3090 반복 실험: 최대 44 MiB, 4.5%). 따라서 다음을 확인한 차이만 환경 효과로 본다.
  - B 또는 C의 재실행이 같은 값을 재현한다.
  - A가 이전 g4dn 값(REF)과 같다.
- OOM이나 오류가 나도 다른 조건으로 바꾸지 않는다. 상태 변화 자체를 결과로 기록한다.
- outside의 주 지표는 `after_warmup_outside`, 보조 지표는 `cuda_init_outside`다.

## 부록 B. (선택) A′ — 데이터센터 R595로 순수 버전 비교

- §3에서 595.112.07이 데이터센터 runfile로 배포되지 않는 것으로 나왔을 때만 한다.
- B를 마친 뒤 데이터센터 R595 최신판을 runfile로 설치하고 재부팅한 다음, 이미지 A로 39개를 다시 실행한다.
  - 버전 확인: https://docs.nvidia.com/datacenter/tesla/
  - 라벨: `aws-g4dn.xlarge__envcmp_Adc`
  - 설치 명령은 §5-B와 같은 runfile 방식이며, 버전만 595.x로 바꾼다. 스크립트는 580만 허용하므로 수동으로 실행한다.

```bash
V=<595.x.y>; curl -fSLO https://us.download.nvidia.com/tesla/$V/NVIDIA-Linux-x86_64-$V.run
sudo systemctl stop nvidia-persistenced; sudo sh NVIDIA-Linux-x86_64-$V.run --silent --dkms --kernel-module-type=<A와 같은 open|proprietary>; sudo reboot
python3 -I tools/run_conditions_env.py --conditions conditions_envcmp39.json --list all --mode measure --gpu-label aws-g4dn.xlarge__envcmp_Adc --attn default --hf-cache ~/hf-cache --image cloudgpufit-bench:cu124-v1 --env-name Adc
```
