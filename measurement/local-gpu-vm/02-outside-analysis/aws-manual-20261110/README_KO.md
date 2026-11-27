# CloudGPUFit AWS 수동 실행 패키지

작성일: 2026-11-10. AWS 인스턴스 생성·기동·실험 실행·중지·결과 회수는 사용자가 직접 수행합니다. 이 패키지에는 유료 인스턴스 생성/시작/자동 중지/Terminate 코드가 없습니다.

## 1. EC2 생성 설정

| 항목 | 설정 |
|---|---|
| 리전 | us-east-1, 미국 동부(버지니아 북부) |
| 인스턴스 | g5.xlarge, A10G 1개, x86_64, On-Demand |
| AMI | 사용자 지정 **ami-0bc11c94e8c4894e4** — Deep Learning OSS NVIDIA Driver AMI GPU PyTorch 2.13 (Ubuntu 26.04), 사용자 설명 기준 |
| 루트 EBS | 150GiB gp3, 3000 IOPS, 125MiB/s (기본 성능) |
| 추가 볼륨 | 없음 |
| 네트워크 | 공인 IPv4 할당, 인터넷/모델 다운로드 가능한 서브넷 |
| 보안 그룹 | SSH TCP22를 접속 PC의 현재 공인IP/32에만 허용 |
| 키 페어 | 사용자가 보유한 PEM 키를 선택 |
| 사용자 데이터 | 비워둠. driver/호스트CUDA를 별도 설치·업데이트하지 않음 |

기존 L4 파일럿도 150GiB급 루트(EBS 파일시스템145G)를 사용했습니다. setup 직전에 최소60GiB 여유를 검사합니다. g5.xlarge의RAM은16GiB이며 작업은4CPU스레드로 제한됩니다.

AMI는 사용자가 실제 선택했다고 알려준 **ami-0bc11c94e8c4894e4**를 기준으로 합니다. 표시 이름과 현재 리전·이미지 상태는 사용자 EC2 콘솔 또는 아래 읽기 전용 조회로 확인합니다. 다른 AMI로 자동 대체하지 않습니다.

```bash
aws ec2 describe-images --region us-east-1 \
  --image-ids ami-0bc11c94e8c4894e4 \
  --query 'Images[].[ImageId,Name,State,Architecture,OwnerId]' --output table
```

선택 리전이 us-east-1이 아니라면 명령과 인스턴스 생성 리전을 실제 리전으로 맞춥니다. AMI ID가 조회되지 않으면 최신 AMI를 임의로 선택하지 말고 리전/ID를 먼저 확인합니다.

기록 충돌: 로컬의 `aws-results/g6.xlarge-20261108T125436Z.tar.gz` 안 `env/instance.txt`에는 `ami-id=ami-0d220f0a0f14edbef`가 적혀 있습니다. 이는 보관된 실행 기록이며, 사용자의 현재 선택 AMI와 동일하다고 단정할 수 없습니다. 원본은 수정하지 않았습니다. 두 기록이 어떤 실행에 해당하는지는 후속 결과의 instance-id/AMI/리전/driver로 구분합니다.

호스트 AMI 이름에 있는 PyTorch/CUDA 버전과 실험용 Docker의 S0/S4 버전은 별개입니다. 이 패키지는 호스트 프레임워크를 사용하지 않고 고정 컨테이너에서 측정합니다. 호스트 driver/runtime 호환성은 setup 및 worker 진단으로 확인하며 driver를 업데이트하지 않습니다.

## 2. ZIP 전송: ZIP과키가있는터미널에서

inugpu에서실행한다면다음경로가실제ZIP경로입니다. 키는해당터미널에서접근할수있는경로여야합니다. 개인키내용을공유할필요는없습니다.

```bash
AWS_KEY=/path/to/your-key.pem
AWS_HOST=YOUR_EC2_PUBLIC_IP
AWS_USER=ubuntu
chmod 400 "$AWS_KEY"
scp -i "$AWS_KEY" /home/inugpu/cloudgpufit-outside/cloudgpufit-aws-manual-20261110.zip "$AWS_USER@$AWS_HOST:~/"
ssh -i "$AWS_KEY" "$AWS_USER@$AWS_HOST"
```

PC에서ZIP을다운로드해전송한다면ZIP경로만PC다운로드경로로바꿉니다. Ubuntu DLAMI기본사용자는ubuntu이며기존AMI를재사용할때EC2의연결탭에서사용자명을확인합니다.

## 3. EC2에서수동시작

드라이버가보이는지만확인합니다. 패키지실행기는기존driver/호스트패키지를업데이트하지않습니다.

```bash
nvidia-smi
python3 -m zipfile -e ~/cloudgpufit-aws-manual-20261110.zip ~/
cd ~/aws-manual-20261110
python3 verify_package.py
nohup sudo -n bash run_all.sh g5.xlarge-01 > run-console.log 2>&1 < /dev/null &
```

기본Ubuntu GPU AMI의sudo권한으로Docker를실행합니다. sudo -n이권한오류로끝나면콘솔로그를확인하고중단하며드라이버를재설치하지않습니다. 원본probeSHA256은고정됩니다. nohup으로SSH연결이끊겨도실행합니다. 같은디렉터리의재시도를막기위해.run-started를남깁니다. 실패한출력을지우고조용히재시도하지않습니다.

실행확인:

```bash
tail -f run-console.log logs/setup.log logs/experiment.log
# Ctrl+C는로그보기를끝낼뿐백그라운드실험을중단하지않음
```

위로그중아직생성안된파일은tail이오류를보일수있습니다. 초기설정은logs/setup.log,본실험은logs/experiment.log를확인하십시오. 최종종료코드:

```bash
cat run.exitcode
ls -lh results/*.tar.gz
```

run.exitcode가아직없으면실행중입니다. 0은setup과전체실행이성공했다는뜻이며최종분석에서하위manifest/품질을다시확인합니다. 원래AWS실행기보다상태/UUID/실제LAZY/케이스검사를강화했습니다. timeout·실패시자동재시도하지않습니다.

## 4. 수행내용

3블록×{S0 torch2.6.0+cu124, S4 torch2.8.0+cu128}×{GPT2,Qwen7B BF16}=12개새프로세스를순차실행합니다. CUDA_MODULE_LOADING=LAZY, eager attention, direct, B1/T256/N64,4CPU threads입니다. direct에도최초forward/generate/warmup/반복추론이포함됩니다.

nvidia-smi가보고한총VRAM이22000MiB미만이면GPT2만선택해6개로줄어들며이사실을기록합니다. 해당분기는호환성필터입니다. 반응을보고측정값을바꾸는보정이아닙니다. worker당timeout1200초,실패시자동중단/원본보존입니다.

Docker이미지는digest고정base와버전고정상위의존성으로다시빌드합니다. 과거와동일imageID/전이의존성이라고가정하지않고imageinspect/pipfreeze/lateCUDA진단을기록합니다. 약15GB모델캐시는EC2에서고정revision으로다운로드하고ZIP에포함하지않습니다. 호스트드라이버나패키지를설치/업데이트하지않습니다.

outside는같은관측구간NVML해당PID−실제torch reserved입니다. xMem/allocated차감이나독립최대값차감은하지않습니다. 프로세스반복은독립GPU환경표본이아닙니다.100MiB기준은practical-outside-criteria.json에보존하며기존criteria.json은변경하지않습니다.

## 5. 결과회수: 다시ZIP을보낸PC/inugpu터미널에서

EC2에서확인한run.exitcode가생성된뒤다운로드합니다.

```bash
mkdir -p aws-results/g5-xlarge-01
scp -i "$AWS_KEY" "$AWS_USER@$AWS_HOST:~/aws-manual-20261110/results/*.tar.gz" aws-results/g5-xlarge-01/
scp -i "$AWS_KEY" "$AWS_USER@$AWS_HOST:~/aws-manual-20261110/run.exitcode" aws-results/g5-xlarge-01/
scp -i "$AWS_KEY" "$AWS_USER@$AWS_HOST:~/aws-manual-20261110/run-console.log" aws-results/g5-xlarge-01/
```

raw 측정 아카이브와 execution 아카이브가 만들어집니다. 전자는 raw/manifest, 후자는 setup/실행 로그·환경·최종 종료 코드입니다. 둘 다 results/*.tar.gz로 회수합니다.

회수한 결과를 본 프로젝트의 aws-results 아래에 두면 그 경로에서 분석을 이어갈 수 있습니다. 개인키나 모델 캐시를 돌려보낼 필요는 없습니다.

## 6. 중지

결과 회수와 파일 존재를 확인한 뒤, AWS EC2 콘솔에서 해당 인스턴스를 Stop합니다. 자동 중지하지 않습니다. Terminate나 EBS 삭제는 결과 보존 확인 후 사용자가 직접 판단하십시오. Stop 후에도 EBS 요금은 남습니다.

OS shutdown은 instance initiated shutdown behavior에 따라 Stop/Terminate가 달라질 수 있어 추가하지 않았습니다.

공식 참고:

- https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-x86-base-gpu-ami-ubuntu-22-04.html
- https://aws.amazon.com/ec2/instance-types/g5/
- https://docs.aws.amazon.com/ebs/latest/userguide/general-purpose.html
- https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Stop_Start.html
