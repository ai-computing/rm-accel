# 빠른 시작 가이드

예제 중심으로 RM Abstract Layer 사용법을 알아봅니다.

---

## 🎯 5분 만에 시작하기

### 1. 가상환경 설정

```bash
# uv 사용 (권장 - 빠름!)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv venv .venv && source .venv/bin/activate

# 또는 venv 사용
python -m venv .venv && source .venv/bin/activate
```

### 2. 설치

```bash
# uv로 설치 (빠름!)
uv pip install -e ".[gpu]"

# 또는 pip 사용
pip install -e ".[gpu]"
```

### 3. 시스템 확인

```bash
python -m rm_abstract.system_validator --quick
```

### 4. 첫 번째 예제

```python
import rm_abstract
from transformers import AutoModelForCausalLM, AutoTokenizer

# 초기화
rm_abstract.init(device="auto")

# 모델 로드 및 추론
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

inputs = tokenizer("Hello, I am", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=30)
print(tokenizer.decode(outputs[0]))
```

---

## 📚 예제 모음

### 예제 1: GPU/vLLM 사용

```python
import rm_abstract
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    # GPU 초기화
    rm_abstract.init(device="gpu:0", verbose=True)

    # 모델 로드
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    # 텍스트 생성
    prompt = "The future of AI is"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True,
    )

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

> ⚠️ **참고**: vLLM 사용 시 `if __name__ == "__main__":` 가드가 필요합니다.

### 예제 2: GPU → CPU 전환

```python
import rm_abstract
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    # GPU로 시작
    rm_abstract.init(device="gpu:0")
    print(f"현재 디바이스: {rm_abstract.get_device_info()}")

    # CPU로 전환
    rm_abstract.switch_device("cpu")
    print(f"전환 후: {rm_abstract.get_device_info()}")

    # CPU에서 추론
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    inputs = tokenizer("Hello", return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0]))
```

### 예제 3: 시스템 정보 확인

```bash
# 터미널에서 실행
python -m rm_abstract.system_info
```

```python
# Python에서 실행
import rm_abstract

# 전체 시스템 정보
rm_abstract.print_system_info()

# 사용 가능한 백엔드
backends = rm_abstract.get_available_backends()
for name, available in backends.items():
    status = "✓" if available else "✗"
    print(f"  {status} {name}")
```

### 예제 4: REST API 서버

**서버 시작:**
```bash
python -m rm_abstract.api --port 8000
```

**API 호출:**
```bash
# 텍스트 생성
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt2", "prompt": "Hello", "max_tokens": 30}'

# 채팅
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt2", "messages": [{"role": "user", "content": "Hi!"}]}'
```

**Python 클라이언트:**
```python
import requests

# 텍스트 생성
response = requests.post(
    "http://localhost:8000/v1/completions",
    json={"model": "gpt2", "prompt": "Hello, I am", "max_tokens": 30}
)
print(response.json()["choices"][0]["text"])
```

### 예제 5: 서빙 엔진 사용

```python
from rm_abstract.serving import (
    create_serving_engine,
    ServingConfig,
    ServingEngineType,
    DeviceTarget,
)

if __name__ == "__main__":
    # vLLM 엔진
    config = ServingConfig(
        engine=ServingEngineType.VLLM,
        device=DeviceTarget.GPU,
    )
    engine = create_serving_engine(config)
    engine.load_model("gpt2")
    output = engine.infer("Hello, I am", max_tokens=30)
    print(output)
```

### 예제 6: 통합 서빙 인터페이스 (자동 서버 관리)

**Context Manager**로 서버 자동 시작/종료:

```python
from rm_abstract.serving import create_serving_engine, ServingConfig, ServingEngineType

if __name__ == "__main__":
    # vLLM (라이브러리 직접 실행)
    config = ServingConfig(engine=ServingEngineType.VLLM)
    with create_serving_engine(config) as engine:
        engine.load_model("gpt2")
        output = engine.infer("Hello, I am", max_tokens=30)
        print(f"vLLM: {output}")
    # with 블록 종료 시 자동 정리

    # Triton (Docker 자동 시작/종료)
    config = ServingConfig(engine=ServingEngineType.TRITON, port=8000)
    with create_serving_engine(config) as engine:
        engine.load_model("gpt2")
        output = engine.infer("Hello, I am")
        print(f"Triton: {output}")
    # Docker 컨테이너 자동 종료

    # TorchServe (서버 프로세스 자동 시작/종료)
    config = ServingConfig(engine=ServingEngineType.TORCHSERVE, port=8080)
    with create_serving_engine(config) as engine:
        engine.load_model("gpt2")
        output = engine.infer("Hello, I am")
        print(f"TorchServe: {output}")
    # 서버 프로세스 자동 종료
```

> 📌 **Triton 사용 시**: 커스텀 Docker 이미지가 필요합니다.
> ```bash
> docker build -t rm-triton-server:latest -f docker/Dockerfile.triton docker/
> ```

---

## 🔧 디바이스 옵션

```python
import rm_abstract

# 자동 선택 (NPU > GPU > CPU)
rm_abstract.init(device="auto")

# 특정 GPU
rm_abstract.init(device="gpu:0")
rm_abstract.init(device="gpu:1")

# CPU
rm_abstract.init(device="cpu")

# Rebellions NPU
rm_abstract.init(device="rbln:0")
```

---

## 📁 예제 파일 실행

### 실행 전 준비

```bash
# 1. 프로젝트 디렉토리로 이동
cd rm_abstract_layer

# 2. 가상환경 활성화
source .venv/bin/activate

# 3. 시스템 확인
python -m rm_abstract.system_validator --quick
```

### 예제 목록

| 파일 | 설명 | 난이도 |
|------|------|--------|
| `basic_usage.py` | 기본 사용법 | ⭐ |
| `device_switching.py` | 디바이스 전환 | ⭐ |
| `gpu_vllm_usage.py` | GPU/vLLM + 디바이스 스위칭 | ⭐⭐ |
| `serving_engines_demo.py` | vLLM, Triton, TorchServe 비교 | ⭐⭐⭐ |
| `unified_serving_demo.py` | **통합 서빙 인터페이스 (자동 서버 관리)** | ⭐⭐⭐ |
| `plugin_system_demo.py` | 플러그인 시스템 데모 | ⭐⭐⭐ |

### 단계별 실행

**1️⃣ 기본 사용법 (초보자용)**
```bash
python examples/basic_usage.py
```

**2️⃣ GPU/vLLM 사용 (GPU 필요)**
```bash
# GPU 확인
nvidia-smi

# 실행
python examples/gpu_vllm_usage.py
```

**3️⃣ 디바이스 전환 테스트**
```bash
python examples/device_switching.py
```

**4️⃣ 서빙 엔진 비교 (고급)**
```bash
# vLLM만 테스트
python examples/serving_engines_demo.py --engine vllm

# 전체 테스트
python examples/serving_engines_demo.py
```

### 실행 옵션

```bash
# 특정 GPU 사용
CUDA_VISIBLE_DEVICES=1 python examples/gpu_vllm_usage.py

# CPU만 사용
python examples/basic_usage.py --device cpu

# 상세 로그 출력
python examples/gpu_vllm_usage.py --verbose
```

### 예상 출력 예시

```
$ python examples/gpu_vllm_usage.py

[INFO] Initializing RM Abstract Layer...
[INFO] Device: gpu:0 (NVIDIA GeForce RTX 3090)
[INFO] Backend: VLLMBackend

Prompt: "The future of AI is"
Output: "The future of AI is bright. With advances in machine learning..."

[INFO] Switching to CPU...
[INFO] Device: cpu:0

Prompt: "Hello, I am"
Output: "Hello, I am a language model trained by..."
```

---

## ❓ 자주 묻는 질문

### Q: 어떤 디바이스가 사용되나요?

```python
import rm_abstract

rm_abstract.init(device="auto")
info = rm_abstract.get_device_info()
print(f"디바이스: {info['device_type']}:{info['device_id']}")
```

### Q: GPU 메모리가 부족해요

```bash
# 다른 GPU 사용
CUDA_VISIBLE_DEVICES=1 python script.py

# 또는 CPU 사용
rm_abstract.init(device="cpu")
```

### Q: 기존 코드를 수정해야 하나요?

아니요! `rm_abstract.init()` 한 줄만 추가하면 됩니다:

```python
import rm_abstract
rm_abstract.init()  # 이 한 줄만 추가

# 기존 코드 그대로 사용
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2")
```

### Q: vLLM 멀티프로세싱 오류가 발생해요

vLLM은 `spawn` 방식의 멀티프로세싱을 사용합니다. 스크립트에 다음을 추가하세요:

```python
if __name__ == "__main__":
    # 코드를 여기에 작성
    main()
```

---

## 🔗 다음 단계

- [INSTALL.md](INSTALL.md) - 상세 설치 가이드 (가상환경 상세 설명)
- [API.md](API.md) - REST API 레퍼런스
- [ARCHITECTURE.md](ARCHITECTURE.md) - 시스템 아키텍처
