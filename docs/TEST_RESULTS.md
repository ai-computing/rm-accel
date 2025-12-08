# 멀티 추론 엔진 GPU/CPU 테스트 결과

## 테스트 환경
- OS: Linux 5.15.0-163-generic
- Python: 3.11
- GPU: 8x NVIDIA GeForce RTX 3090
- vLLM: v0.12.0 (V1 Engine)
- transformers: 4.57.3
- torch: 2.9.0+cu128

## 테스트 결과 요약

### ✅ 성공한 테스트 (3/5)

| Test | Device | Engine | Hook | Result |
|------|--------|--------|------|--------|
| CPU with vLLM hook | CPU | vllm (fallback) | Enabled | ✅ SUCCESS |
| CPU transformers | CPU | transformers | Disabled | ✅ SUCCESS |
| Auto-import CPU | CPU | vllm (fallback) | Enabled | ✅ SUCCESS |
| GPU with vLLM | GPU | vllm | Disabled | ❌ HANG |
| Auto-import GPU | GPU | vllm | Disabled | ❌ NOT TESTED |

### ✅ 성공 사례

#### 1. CPU with vLLM Hook
```python
from rm_abstract.auto import use
use(device="cpu", engine="vllm", hook_vllm=True)

from rm_abstract import LLM, SamplingParams
llm = LLM(model="gpt2", device="cpu", engine="vllm")
outputs = llm.generate(["Hello, I am"], SamplingParams(max_tokens=20))
```

**결과**: ✅ 성공
- vLLM 후킹이 정상 작동
- CPU에서 transformers로 정상 폴백
- 추론 결과 정상 출력

#### 2. CPU with transformers directly
```python
use(device="cpu", engine="vllm", hook_vllm=False)
llm = LLM(model="gpt2", device="cpu")
```

**결과**: ✅ 성공
- transformers 직접 사용 정상 작동
- 후킹 비활성화 확인

#### 3. Auto-import with CPU
```python
from rm_abstract.auto import use
use(device="cpu", engine="vllm", hook_vllm=True)

from vllm import LLM, SamplingParams  # Auto-hooked
llm = LLM(model="gpt2")
```

**결과**: ✅ 성공
- `from vllm import LLM` 자동 후킹 성공
- compat 모듈로 정상 라우팅

## 발견한 문제점

### ❌ Issue #1: vLLM V1 Engine Hang on GPU

**증상**:
- vLLM V1 엔진이 GPU에서 초기화 후 멈춤 (hang)
- EngineCore 프로세스가 parallel_state 초기화 후 진행 안 됨
- 5분 이상 대기해도 응답 없음

**로그**:
```
[EngineCore_DP0 pid=49300] INFO 12-08 11:37:31 [core.py:93] Initializing a V1 LLM engine (v0.12.0)
[EngineCore_DP0 pid=49300] INFO 12-08 11:37:32 [parallel_state.py:1200] world_size=1 rank=0 local_rank=0
[EngineCore_DP0 pid=49300] INFO 12-08 11:37:33 [parallel_state.py:1408] rank 0 assigned as DP rank 0
(이후 진행 없음)
```

**환경**:
- vLLM 0.12.0 (V1 Engine)
- enable_prefix_caching=True
- enable_chunked_prefill=True
- enforce_eager=True

**추정 원인**:
1. vLLM V1 엔진의 버그 또는 설정 문제
2. NCCL backend 초기화 문제
3. GPU 메모리 할당 데드락

**시도한 해결책**:
- ✅ `enforce_eager=True` 추가 (컴파일 비활성화)
- ✅ `CUDA_VISIBLE_DEVICES=1` 설정 (GPU 변경)
- ❌ 여전히 hang 발생

### ❌ Issue #2: GPU 메모리 경합

**증상**:
- TorchServe가 GPU 0에서 ~19GB 메모리 사용
- vLLM 초기화 시 GPU 0 선택하면 메모리 부족 가능성

**해결책**:
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use different GPU
```

### ⚠️ Issue #3: vLLM 컴파일 모드 성능 문제

**증상**:
- vLLM 기본 설정(컴파일 모드)에서 초기화 매우 느림
- VLLM_COMPILE 모드에서 수십 분 소요

**해결책**:
```python
llm = LLM(model="gpt2", enforce_eager=True)
```

## 코드 수정 사항

### 1. auto.py - GPU/NPU에서 후킹 자동 비활성화

**변경 전**:
```python
_config = {
    "hook_vllm": os.environ.get("RM_HOOK_VLLM", "1") == "1",
}
```

**변경 후**:
```python
def use(device=None, engine=None, hook_vllm=None):
    resolved_device = device or _config["device"]
    if resolved_device in ("nvidia", "gpu", "cuda", "rbln", "furiosa"):
        # For GPU/NPU, disable hooking - use original vllm
        if hook_vllm is None:
            hook_vllm = False
```

**이유**: GPU/NPU에서는 원본 vllm을 사용하고, CPU에서만 transformers로 폴백

### 2. test_multi_engine.py - 테스트 스크립트 개선

**추가**:
- GPU 장치 명시적 선택 (`CUDA_VISIBLE_DEVICES`)
- vLLM eager 모드 사용 (`enforce_eager=True`)
- 로그 비활성화 (`disable_log_stats=True`)

## 권장 사항

### 1. GPU 사용 시

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Select specific GPU

from rm_abstract import LLM, SamplingParams
llm = LLM(
    model="gpt2",
    device="nvidia",
    enforce_eager=True,  # Disable compilation for faster init
    gpu_memory_utilization=0.15,  # Low memory usage for shared environments
)
```

### 2. CPU 사용 시

```python
from rm_abstract.auto import use
use(device="cpu", hook_vllm=True)

from vllm import LLM  # Auto-hooked to transformers
llm = LLM(model="gpt2")
```

### 3. vLLM V1 Engine 문제 우회

현재 vLLM V1 엔진이 hang하는 문제가 있으므로, 프로덕션 환경에서는:

1. **vLLM V0 엔진 사용** (환경변수로 제어 가능)
2. **transformers 직접 사용** (CPU 또는 GPU)
3. **다른 serving 엔진 사용** (Triton, TorchServe)

## 다음 단계

1. ❗ **vLLM V1 Engine hang 문제 해결**
   - vLLM 버전 업그레이드 또는 다운그레이드
   - V0 엔진으로 fallback 옵션 추가
   - enable_prefix_caching, enable_chunked_prefill 옵션 조사

2. **GPU 메모리 관리 개선**
   - 자동 GPU 선택 로직 추가
   - 메모리 부족 시 자동 폴백

3. **통합 테스트 추가**
   - Triton engine 테스트
   - TorchServe engine 테스트
   - Multi-GPU 테스트

## 결론

### 작동하는 기능 ✅
- **CPU 추론** (transformers fallback)
- **vLLM 후킹** (자동 import 변환)
- **Auto-configuration** (device 기반 후킹 제어)

### 현재 문제 ❌
- **vLLM V1 Engine GPU hang** (critical)
- GPU 메모리 경합 (workaround 있음)

### 개선 완료 ✅
- GPU/NPU에서 자동 후킹 비활성화
- vLLM 컴파일 모드 비활성화 옵션

**전체 평가**: CPU는 완벽히 작동하나, GPU는 vLLM V1 엔진 문제로 추가 작업 필요
