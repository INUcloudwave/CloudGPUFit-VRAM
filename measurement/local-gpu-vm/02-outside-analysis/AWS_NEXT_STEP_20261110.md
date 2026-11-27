> AMI 정정(사용자 최신 확인): 앞으로 사용할 AMI는 `ami-0bc11c94e8c4894e4`입니다. 사용자가 알려준 표시 이름은 Deep Learning OSS NVIDIA Driver AMI GPU PyTorch 2.13 (Ubuntu 26.04)입니다. 현재 공개 여부·표시 이름·리전은 여기서 AWS API로 검증하지 않았습니다. 공식 Ubuntu22.04 이미지로 임의 대체하지 않습니다.
> 보관된 L4 tar의 env/instance.txt에는 `ami-0d220f0a0f14edbef`가 있으나 이것을 사용자의 현재 AMI라고 단정한 이전 안내를 철회합니다. 원시 기록은 보존하고 실행별로 구분합니다.

> 최신 운영 지시(2026-11-10): AWS 인스턴스 생성과 실험 실행은 사용자가 직접 합니다. 에이전트는 AWS에 접속하거나 실험을 실행하지 않고, 코드 제공과 회수된 결과 분석만 수행합니다.
> 수동 실행 ZIP: cloudgpufit-aws-manual-20261110.zip
> 상세 명령: aws-manual-20261110/README_KO.md

# 로컬 마무리 후 AWS 다음 실행

2026-11-10. 사용자가 AWS 인스턴스를 직접 준비하기로 했습니다. 이 환경에는 AWS CLI가 없고 접속 대상은 아직 제공되지 않았습니다. 유료 자원을 생성/시작/중지/종료하지 않았습니다.

## 다음 표본

다음은 g5.xlarge(A10G) 단일GPU가 적합합니다. 기존 g6.xlarge(L4) 파일럿은 성공했으므로 같은파일럿을반복하기보다GPU환경표본을추가합니다. AWS G5공식정보: https://aws.amazon.com/ec2/instance-types/g5/

목적은 GPU별 outside의반복안정성과 S0/S4 stack민감도를같이기록하는것입니다. g5와3090 비교만으로driver효과를분리했다고주장하지않습니다. 실제driver/AMI/imageID/실측CUDA/runtime/모델revision/입력hash/GPU UUID를대조해야합니다. 기존L4실행리전은us-east-1이며같은리전/AMI를쓸지사용자가정합니다. 새AMI의현재유효성·가격·쿼터·기동가능여부는여기서검증하지않았습니다.

## 준비물 점검 결과

- 기존 aws-bundle.tar.gz 내부10개파일은 aws-bundle/과바이트내용일치.
- lab/probe.py SHA256: 7df9a02ce9ca938b9702ee2c9fd7d0414d0a4cfc0cfe5ab87c97d7a069243d88.
- Python실행파일구문검사통과. S0/S4 base이미지는digest고정.
- S0/S4는레시피로다시빌드하므로과거L4/로컬과동일imageID라고가정하지않습니다. 실제imageinspect와pipfreeze/lateCUDA진단을기록합니다. 최상위의존성을고정해도전이의존성은달라질수있으므로manifest에서비교합니다.
- 기본스케줄:3블록×S0/S4×GPT2/Qwen=12 fresh worker. nvidia-smi보고총량22000MiB미만이면setup이GPT2만선택하여6회입니다. 메모리크기는호환성필터이며outside예측입력으로쓴것이아닙니다.
- 새측정반복을GPU환경반복으로세지않습니다. 새GPU미측정예측은이측정값조회와다르며아직검증전입니다.

## 인스턴스가 준비되면

우선SSH주소·사용자명·키파일경로또는SSM접속방법을제공하면에이전트가직접전송/실행할수있습니다. 개인키내용을채팅에붙여넣지않습니다. 로컬15GB캐시는복제하지않고기존번들을전송해대상에서고정revision을준비합니다.

기존 번들 자체는 변경하지 않았습니다. 인스턴스에서 다음 두 단계로 실행합니다.

```bash
# 번들을 전송·풀어둔 aws-bundle 디렉터리에서
./setup.sh
```

```bash
# setup 성공 및 nvidia-smi/이미지/계획 확인 후, 지속 세션에서
python3 run.py --label g5.xlarge-01 > g5.xlarge-01.log 2>&1
aws_run_exit=$?
printf '%s\n' "$aws_run_exit" > g5.xlarge-01.exitcode
```

접속을끊어도실행되도록tmux나동등한지속수단을사용합니다. 완료/실패시results/*.tar.gz와로그·종료코드를회수한후orchestration/하위manifest/exitcode를확인합니다. 실패를자동재시도하지않습니다.

## 중지 절차에 대한 정정

기존NOTION_SUMMARY에는실행후자동shutdown이라고적혀있었으나현재setup.sh/run.py에는shutdown/StopInstances호출이없습니다. 이번점검에서이를확인했습니다. 기존번들이자동중지한다고기대하면안됩니다.

완료후결과를회수하고AWS콘솔에서해당인스턴스의Stop을명시적으로수행하는절차가필요합니다. OS shutdown은인스턴스설정에따라Stop또는Terminate가될수있으므로설정확인없이자동삽입하지않았습니다. AWS도사용자중지는콘솔/API/CLI를권장합니다.

- https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Stop_Start.html
- https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Using_ChangingInstanceInitiatedShutdownBehavior.html

Stop후에도EBS스토리지요금은남습니다. 결과보존확인후불필요한인스턴스/볼륨정리는사용자가승인한범위에서별도로처리합니다. 자동Terminate/볼륨삭제는추가하지않았습니다.

현재경계: 인스턴스준비와SSH/SSM접속정보가필요합니다. 접속정보가오면기동된대상에대해번들전송·사전점검·실험을이어갈수있습니다. 유료인스턴스자동기동금지라는최초운영규칙과사용자가직접준비한다는최신지시를유지합니다.
