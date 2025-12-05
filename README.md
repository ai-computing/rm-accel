# RM Abstract Layer

> **GPU / NPU / CPU 이기종 가속기를 통합 관리하는 LLM 추론 추상화 레이어**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 개요

RM Abstract Layer는 다양한 하드웨어 가속기(GPU, NPU, CPU)에서 LLM 추론을 **코드 수정 없이** 실행할 수 있게 해주는 추상화 레이어입니다.

### 주요 특징

- 🔄 **디바이스 투명성**: 기존 HuggingFace 코드가 그대로 동작
- ⚡ **런타임 스위칭**: GPU ↔ CPU ↔ NPU 실시간 전환
- 🚀 **다중 서빙 엔진**: vLLM, Triton, TorchServe 지원
- 🔌 **플러그인 아키텍처**: 새로운 백엔드 쉽게 추가
- 🌐 **REST API**: OpenAI 호환 API 서버

---

## ⚡ 빠른 시작

### 설치

```bash
# 기본 설치
pip install -e .

# GPU 지원
pip install -e ".[gpu]"

# 전체 설치
pip install -e ".[all]"
```

### 시스템 확인

```bash
# 시스템 검증 (실제 테스트)
python -m rm_abstract.system_validator

# 설치 상태 확인
python -m rm_abstract.installer
```

### 기본 사용법

```python
import rm_abstract
from transformers import AutoModelForCausalLM, AutoTokenizer

# 초기화 (자동으로 최적 디바이스 선택)
rm_abstract.init(device="auto")

# 기존 HuggingFace 코드 그대로 사용
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# 추론
inputs = tokenizer("Hello, I am", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```

### 디바이스 전환

```python
import rm_abstract

# GPU로 시작
rm_abstract.init(device="gpu:0")

# CPU로 전환
rm_abstract.switch_device("cpu")

# 현재 디바이스 확인
info = rm_abstract.get_device_info()
print(f"현재: {info['device_type']}:{info['device_id']}")
```

---

## 🏗️ 지원 환경

### 하드웨어

| 디바이스 | 상태 | 백엔드 |
|----------|------|--------|
| NVIDIA GPU | ✅ 지원 | vLLM |
| CPU | ✅ 지원 | PyTorch |
| Rebellions ATOM NPU | ✅ 지원 | vLLM-RBLN / Optimum-RBLN |
| FuriosaAI NPU | 🔄 계획 | - |

### 서빙 엔진

| 엔진 | 상태 | 특징 | 실행 방식 |
|------|------|------|----------|
| vLLM | ✅ 테스트 완료 | 고성능 LLM 서빙 | 라이브러리 직접 실행 |
| Triton | ✅ 테스트 완료 | 다중 모델 서빙 | Docker 컨테이너 자동 관리 |
| TorchServe | ✅ 테스트 완료 | PyTorch 네이티브 | Java 서버 자동 관리 |

---

## 🔄 통합 서빙 인터페이스

모든 서빙 엔진을 **동일한 인터페이스**로 사용할 수 있습니다:

```python
from rm_abstract.serving import create_serving_engine, ServingConfig, ServingEngineType

# vLLM 사용
config = ServingConfig(engine=ServingEngineType.VLLM, model_name="gpt2")
with create_serving_engine(config) as engine:
    engine.load_model("gpt2")
    output = engine.infer("Hello, I am")
    print(output)

# Triton 사용 (Docker 자동 시작/종료)
config = ServingConfig(engine=ServingEngineType.TRITON, port=8000)
with create_serving_engine(config) as engine:
    engine.load_model("gpt2")
    output = engine.infer("Hello, I am")
    print(output)

# TorchServe 사용 (서버 자동 시작/종료)
config = ServingConfig(engine=ServingEngineType.TORCHSERVE, port=8080)
with create_serving_engine(config) as engine:
    engine.load_model("gpt2")
    output = engine.infer("Hello, I am")
    print(output)
```

### Context Manager 지원

- `with` 블록 진입 시 서버 자동 시작
- `with` 블록 종료 시 서버 자동 정리
- 예외 발생 시에도 안전하게 정리

---

## 🌐 REST API 서버

OpenAI 호환 API 서버 제공:

```bash
# 서버 시작
python -m rm_abstract.api --port 8000

# API 문서
open http://localhost:8000/docs
```

```bash
# 텍스트 생성
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt2", "prompt": "Hello", "max_tokens": 50}'

# 채팅
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt2", "messages": [{"role": "user", "content": "Hi!"}]}'
```

---

## 📊 시스템 검증

```bash
python -m rm_abstract.system_validator
```

```
======================================================================
  RM Abstract Layer - System Validation
======================================================================

  Testing GPU Available... ✓
  Testing CPU Inference... ✓
  Testing vLLM GPU Inference... ✓
  Testing Device Switching... ✓
  Testing Triton... ✓
  Testing TorchServe... ✓

Summary:
  ✅ Passed:   6
  ❌ Failed:   0
======================================================================
```

---

## 📚 문서

| 문서 | 설명 |
|------|------|
| [INSTALL.md](INSTALL.md) | 상세 설치 가이드 |
| [QUICKSTART.md](QUICKSTART.md) | 예제 중심 빠른 시작 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 시스템 아키텍처 |
| [API.md](API.md) | REST API 레퍼런스 |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 개발/기여 가이드 |

---

## 🛠️ 예제

```
examples/
├── basic_usage.py           # 기본 사용법
├── device_switching.py      # 디바이스 전환
├── gpu_vllm_usage.py        # GPU/vLLM 사용 예제
├── serving_engines_demo.py  # 서빙 엔진 비교
└── unified_serving_demo.py  # 통합 서빙 인터페이스 (자동 서버 관리)
```

```bash
# 기본 사용법
python examples/basic_usage.py

# GPU/vLLM 예제 실행
python examples/gpu_vllm_usage.py

# 통합 서빙 데모 (권장)
python examples/unified_serving_demo.py

# 특정 엔진으로 서빙 데모
python examples/serving_engines_demo.py --engine vllm
python examples/serving_engines_demo.py --engine triton
python examples/serving_engines_demo.py --engine torchserve
```

---

## 🧪 테스트

```bash
# 전체 테스트
pytest tests/ -v

# 코어 테스트만
pytest tests/test_core.py -v

# API 테스트만
pytest tests/test_api.py -v
```

---

## 📦 프로젝트 구조

```
rm_abstract_layer/
├── src/rm_abstract/
│   ├── api/              # REST API 서버 (OpenAI 호환)
│   ├── backends/         # 백엔드 구현
│   │   ├── cpu/          # CPU 백엔드 (PyTorch)
│   │   ├── gpu/          # GPU 백엔드 (vLLM)
│   │   └── npu/          # NPU 백엔드 (Rebellions RBLN)
│   ├── serving/          # 서빙 엔진 (통합 인터페이스)
│   │   ├── vllm_engine.py      # vLLM (라이브러리)
│   │   ├── triton_engine.py    # Triton (Docker)
│   │   └── torchserve_engine.py # TorchServe (Java)
│   ├── core/             # 코어 모듈
│   ├── system_info.py    # 시스템 정보
│   ├── system_validator.py # 시스템 검증
│   └── installer.py      # 설치 헬퍼
├── tests/                # pytest 테스트
├── examples/             # 예제 스크립트
├── requirements/         # 의존성 파일
├── scripts/              # 설치 스크립트
└── docker/               # Docker 설정
    └── Dockerfile.triton # Triton 커스텀 이미지
```

---

## 📄 라이선스

MIT License

---

## 🤝 기여

기여를 환영합니다! [CONTRIBUTING.md](CONTRIBUTING.md)를 참조해주세요.
