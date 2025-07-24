# TinyLlama 타로 증류 파이프라인

- 로컬에서 TinyLlama를 타로 응답에 특화되도록 증류(distillation)하고, 학습된 모델을 GGUF 포맷으로 변환하여 WebLLM 등에서 사용할 수 있도록 구성된 파이프라인
- 아직 임시 gpt 버전

## 📁 디렉토리 구조

```
.
├── config.json          # 전체 설정값
├── distillation.py      # 학습 및 GGUF 변환 스크립트
├── evaluate.py          # test 데이터로 예측 결과 평가
├── index.py             # 전체 파이프라인 실행 CLI
├── makePrompt.py        # promptOrigin → promptTrain/Test 생성
├── ouputGguf/           # 변환된 gguf 파일 저장
├── promptOrigin/        # 원본 하루치 대화 (.txt)
├── promptTrain/         # 학습용 jsonl (하루치 전체 대화 한 줄 / 프롬프트 + expected)
└── llama.cpp/           # GGUF 변환용 스크립트가 들어있는 디렉토리 (경로 주의)
```

> ⚠️ **주의**: `llama.cpp` 디렉토리는 이 프로젝트 기준 **상위 폴더의 `llama.cpp` 폴더**에 위치해야 합니다. 즉 `../llama.cpp` 위치에 있어야 하며, `convert.py`는 `../llama.cpp/convert.py`로 접근됩니다.

---

## 🛠 설치

Python 3.9+ 환경을 권장하며, 가상환경 사용을 추천합니다.

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

필요한 주요 패키지:

- `transformers`
- `datasets`
- `accelerate`
- `torch` (CPU/MPS/CUDA 환경에 맞춰 설치)

---

## ⚙️ 실행 방법

### 전체 파이프라인 실행

```bash
# index.py 실행 예시 (CLI)
python index.py make            # makePrompt만 실행
python index.py distill         # distillation만 실행
python index.py all             # 전체 파이프라인 실행
```

## 🧠 모델 초기 다운로드

모델이 로컬에 없을 경우 `./models/` 경로로 자동 다운로드되며, 이후부터는 해당 디렉토리의 로컬 모델만 사용됩니다.

---

## 🧾 .gitignore 예시

```gitignore
__pycache__/
*.pyc
.venv/
ouputGguf/
models/
*.gguf
*.jsonl
```

---

## 📦 GGUF 변환 실패 시 확인할 점

- `llama.cpp/convert.py`가 `../llama.cpp/convert.py`에 정확히 위치했는지 확인
- Windows에서도 `subprocess.run([...])` 실행 시 `python` 경로가 제대로 연결됐는지 확인 (필요 시 `"python"` → `"python3"` 혹은 `sys.executable` 사용)
- `PYTORCH_MPS_HIGH_WATERMARK_RATIO`는 Windows에서는 무관함 (Mac 전용)
