# CPU 검증 이미지 기록 (cloudgpufit-cpu-validation:20261108)

- 이미지 ID: `sha256:875290b5854ad6dbc8ea554fecb292dd5ffdfea2a03e8c4e8179a3c827968fa5`
  - 2026-11-08 생성, linux/amd64, 레이어 6개
- 내용: Ubuntu 22.04.5, Python 3.11.11, torch 2.6.0+cu124, transformers 4.52.4
  - 전체 목록: `pip_freeze_validation_image.txt`, 154개
- 만든 방법 (레이어 기록으로 추정)
  - 베이스 `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime` (`@sha256:77f17f84…`, INUOS에 있음)에 pip 패키지 43개(`pip_added_over_base.txt`)를 설치한 컨테이너를 커밋한 로컬 이미지다.
  - 마지막 레이어 명령은 `sleep infinity`다.
- 이 이미지는 INUOS 도커에 그대로 남아 있다. 삭제한 것은 AWS로 보낸 압축 파일뿐이다.

## 다시 필요할 때

1. **같은 파일 다시 만들기 (권장)**: `bash docs/image/export_image.sh <출력 폴더>`
   - 2026-11-17 확인: 다시 내보낸 결과의 sha256이 AWS로 보낸 번들 파일과 같다.
   - sha256 `8642784f94181ff45045b04cc61fd772b4f70a387ef65ce1167c1532f8ef1897`
2. **로컬 이미지까지 잃었을 때만**: `docker build -f docs/image/Dockerfile.rebuild -t cloudgpufit-cpu-validation:rebuild docs/image`
   - 베이스 digest와 패키지 버전이 같은 동등한 환경이 만들어진다.
   - 이미지 ID는 달라지므로 결과마다 새 ID를 기록한다.
   - `ures==1.9.1` 줄은 pip freeze 출력을 그대로 옮긴 것이다. 설치가 실패하면 이 줄만 확인한다.
