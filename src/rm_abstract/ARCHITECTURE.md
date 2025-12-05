# RM Abstract Layer - 내부 아키텍처

## 핵심 개념

### Backend vs ServingEngine

두 개념은 다른 목적을 가집니다:

```
┌─────────────────────────────────────────────────────────────┐
│                    사용자 코드                               │
│  model = AutoModelForCausalLM.from_pretrained("gpt2")       │
│  output = model.generate(...)                               │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│    Backend      │  │    Backend      │  │    Backend      │
│   (CPU/PyTorch) │  │   (GPU/vLLM)    │  │   (NPU/RBLN)    │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                    하드웨어 추상화
```

```
┌─────────────────────────────────────────────────────────────┐
│                    서빙 클라이언트                            │
│  curl http://localhost:8000/v1/completions                  │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  ServingEngine  │  │  ServingEngine  │  │  ServingEngine  │
│     (vLLM)      │  │    (Triton)     │  │  (TorchServe)   │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                    서빙 인프라 추상화
```

## Backend (하드웨어 추상화)

**위치**: `rm_abstract/backends/`

**역할**:
- 특정 하드웨어에서 모델을 실행하는 방법 정의
- HuggingFace 코드 후킹을 통한 투명한 디바이스 전환
- 모델 준비 (컴파일, 최적화)
- 추론 실행

**구현체**:
| Backend | 하드웨어 | 라이브러리 |
|---------|----------|-----------|
| `CPUBackend` | CPU | PyTorch |
| `VLLMBackend` | NVIDIA GPU | vLLM |
| `RBLNBackend` | Rebellions NPU | vLLM-RBLN / Optimum-RBLN |
| `FuriosaBackend` | FuriosaAI NPU | furiosa-sdk |

**사용 패턴**:
```python
import rm_abstract

# 백엔드 자동 선택 및 초기화
rm_abstract.init(device="gpu:0")

# 기존 코드가 자동으로 해당 백엔드에서 실행됨
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2")
output = model.generate(...)  # GPU/vLLM에서 실행
```

## ServingEngine (서빙 인프라 추상화)

**위치**: `rm_abstract/serving/`

**역할**:
- 모델 서빙 서버 관리 (시작, 중지, 헬스체크)
- HTTP/gRPC API 제공
- 배치 처리, 로드 밸런싱
- 모델 버저닝

**구현체**:
| Engine | 실행 방식 | 특징 |
|--------|----------|------|
| `VLLMServingEngine` | 라이브러리 | 고성능, PagedAttention |
| `TritonServingEngine` | Docker | 다중 모델, 앙상블 |
| `TorchServeEngine` | Java 서버 | PyTorch 네이티브 |

**사용 패턴**:
```python
from rm_abstract.serving import create_serving_engine, ServingConfig, ServingEngineType

config = ServingConfig(engine=ServingEngineType.VLLM, port=8000)

with create_serving_engine(config) as engine:
    engine.load_model("gpt2")
    output = engine.infer("Hello, I am")
    # HTTP API: http://localhost:8000/v1/completions
```

## 둘의 관계

```
┌─────────────────────────────────────────────────────────────┐
│                     ServingEngine                           │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                     Backend                         │   │
│   │  (실제 추론 실행)                                    │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
│   + 서버 라이프사이클 관리                                   │
│   + HTTP/gRPC API                                          │
│   + 배칭, 스케줄링                                          │
└─────────────────────────────────────────────────────────────┘
```

- `Backend`는 단일 추론 요청 처리에 집중
- `ServingEngine`은 Backend를 감싸서 서비스로 제공

## 언제 무엇을 사용?

| 시나리오 | 권장 |
|---------|------|
| 로컬 개발/테스트 | `rm_abstract.init()` + Backend |
| 프로덕션 배포 | `ServingEngine` |
| Jupyter 노트북 | `rm_abstract.init()` + Backend |
| REST API 서버 | `ServingEngine` 또는 `rm_abstract.api` |
| 디바이스 전환 실험 | `rm_abstract.init()` + `switch_device()` |

## 디렉토리 구조

```
src/rm_abstract/
├── core/
│   ├── backend.py          # Backend 추상 클래스
│   ├── controller.py       # 디바이스 플로우 컨트롤러
│   └── config.py           # 설정
├── backends/
│   ├── cpu/                # CPU Backend
│   ├── gpu/                # GPU Backend (vLLM)
│   └── npu/                # NPU Backends (RBLN, Furiosa)
├── serving/
│   ├── base.py             # ServingEngine 추상 클래스
│   ├── vllm_engine.py      # vLLM 서빙
│   ├── triton_engine.py    # Triton 서빙
│   └── torchserve_engine.py # TorchServe 서빙
└── api/                    # REST API 서버
```

