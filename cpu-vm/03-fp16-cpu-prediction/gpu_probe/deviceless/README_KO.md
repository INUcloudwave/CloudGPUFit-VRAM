# cuDNN 버전 영향 + deviceless 확인 (2026-11-23)

- `run_version_test.sh` (INUGPU 3090에서 실행)
  - 같은 torch 2.6 이미지에서 cuDNN만 바꿔 끼운다(LD_PRELOAD): 9.1.0 / 9.5.1 / 9.7.1 / 9.10.2 / 9.12.0 / 9.28.0
  - 4,206개 형태(`conv_signatures_all.json`)를 기존 probe로 측정한다.
  - PyPI 휠은 공식 sha256(`wheels_SHA256SUMS`)으로 확인한다.
- `deviceless_query.py`
  - 직렬화한 GPU 속성만으로(핸들 없이) conv_fprop 그래프를 만든다.
  - 휴리스틱(A, 그다음 FALLBACK)의 모든 후보 계획에 대해 이름, 빌드 성공 여부, 작업 공간을 기록한다.
- INUOS(CPU, GPU·드라이버 없음)에서 확인한 것
  - torch 2.6에 cuDNN 9.28을 바꿔 끼우면 `torch.backends.cudnn.version()` = 92800이 나온다.
  - frontend 1.32.2 API: `create_device_properties(str)`, `get_execution_plan_count`, `get_workspace_size_plan_at_index` 등
  - `libcudnn_engines_runtime_compiled`가 libcuda.so.1을 요구한다.
    - conda-forge `cuda-driver-dev_linux-64 12.4.127`의 stub로 해결했다(`third_party/deviceless/stub`, 패키지 sha256 78f2e2c7…).
  - stub 상태에서 속성 JSON 읽기 단계까지 동작한다. 실제 조회에는 GPU에서 뽑은 속성 파일이 필요하다.
- 로컬 설치: `third_party/deviceless/site` (cudnn 9.28.0.42, frontend 1.32.2. git 제외)
