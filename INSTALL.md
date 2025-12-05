# 설치 가이드

RM Abstract Layer 설치 및 환경 설정 가이드입니다.

---

## 📋 요구사항

### 시스템 요구사항

- **Python**: 3.9 이상
- **OS**: Linux (Ubuntu 20.04+), macOS
- **메모리**: 최소 8GB RAM (16GB 권장)

### 하드웨어별 요구사항

| 하드웨어 | 요구사항 |
|----------|----------|
| GPU | NVIDIA GPU (CUDA 11.8+), 8GB+ VRAM |
| NPU (RBLN) | Rebellions ATOM + RBLN SDK |
| CPU | x86_64 또는 ARM64 |

---

## 🐍 가상환경 설정

패키지 충돌을 방지하기 위해 **가상환경 사용을 강력히 권장**합니다.

### 방법 1: uv (권장) ⚡

[uv](https://github.com/astral-sh/uv)는 Rust로 작성된 초고속 Python 패키지 매니저입니다.

**장점:**
- pip보다 10-100배 빠른 설치 속도
- 자동 Python 버전 관리
- 빌트인 가상환경 지원
- pip/venv 완벽 호환

```bash
# uv 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 터미널 재시작 또는
source ~/.bashrc  # 또는 ~/.zshrc

# 가상환경 생성
uv venv .venv

# 활성화
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate      # Windows

# 패키지 설치 (초고속!)
uv pip install -e ".[all]"
```

### 방법 2: venv (Python 기본)

Python 3.3+ 내장 가상환경 도구입니다.

```bash
# 가상환경 생성
python -m venv .venv

# 활성화
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate      # Windows

# 패키지 설치
pip install -e ".[all]"
```

### 방법 3: conda

Anaconda/Miniconda 사용자를 위한 방법입니다.

```bash
# 환경 생성
conda create -n rm_abstract python=3.10

# 활성화
conda activate rm_abstract

# 패키지 설치
pip install -e ".[all]"
```

### 가상환경 비활성화

```bash
deactivate  # venv, uv
conda deactivate  # conda
```

---

## 🚀 빠른 설치

### 기본 설치

```bash
# 가상환경 활성화 후
pip install -e .

# GPU 지원
pip install -e ".[gpu]"

# 전체 설치
pip install -e ".[all]"
```

### uv로 빠른 설치 (권장)

```bash
# 한 번에 설정 (uv 설치 → 가상환경 → 패키지)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[all]"
```

---

## 📦 컴포넌트별 설치

### 설치 상태 확인

```bash
python -m rm_abstract.installer
```

출력 예시:
```
============================================================
  RM Abstract Layer - Installation Guide
============================================================

Components:
  ✓ Base: Core functionality
  ✓ GPU (vLLM): High-performance GPU inference
  ✓ Triton: Multi-model serving
  ✓ TorchServe: PyTorch native serving

System Dependencies:
  ✗ Java 11: Required for TorchServe server
  ✓ Docker: Required for Triton server
```

### Python 패키지 설치

```bash
# 컴포넌트별 설치
python -m rm_abstract.installer base        # 기본
python -m rm_abstract.installer gpu         # GPU/vLLM
python -m rm_abstract.installer triton      # Triton
python -m rm_abstract.installer torchserve  # TorchServe
python -m rm_abstract.installer all         # 전체
```

### 시스템 의존성 설치

```bash
# 스크립트 사용
./scripts/install_deps.sh java              # Java (TorchServe용)
./scripts/install_deps.sh docker            # Docker (Triton용)
./scripts/install_deps.sh nvidia-docker     # NVIDIA Container Toolkit
```

#### 수동 설치

**Java 11 (TorchServe용)**
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install -y openjdk-11-jdk

# RHEL/CentOS
sudo yum install -y java-11-openjdk

# macOS
brew install openjdk@11
```

**Docker (Triton용)**
```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

---

## 🔧 상세 설치

### GPU (vLLM) 설치

```bash
# requirements 파일 사용
pip install -r requirements/gpu.txt

# 또는 직접 설치
pip install vllm>=0.4.0 torch>=2.0.0
```

**확인:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

import vllm
print(f"vLLM version: {vllm.__version__}")
```

### Triton 설치

```bash
# 클라이언트 설치
pip install -r requirements/triton.txt

# 커스텀 Docker 이미지 빌드 (권장 - transformers 포함)
docker build -t rm-triton-server:latest -f docker/Dockerfile.triton docker/
```

> 📌 **커스텀 이미지 필요 이유**: 기본 Triton 이미지에는 `transformers`가 없어서 
> Python 백엔드로 LLM 모델을 로드할 수 없습니다.

**서버 시작 (자동):**
```python
# 통합 인터페이스 사용 시 자동으로 Docker가 시작/종료됨
from rm_abstract.serving import create_serving_engine, ServingConfig, ServingEngineType

with create_serving_engine(ServingConfig(engine=ServingEngineType.TRITON)) as engine:
    engine.load_model("gpt2")
    output = engine.infer("Hello")
```

**서버 시작 (수동):**
```bash
# Docker Compose 사용
docker-compose -f docker/docker-compose.yml up triton

# 또는 직접 실행
docker run --gpus all --rm -d \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v ~/.rm_abstract/triton_models:/models \
  rm-triton-server:latest \
  --model-repository=/models
```

### TorchServe 설치

```bash
# 패키지 설치
pip install -r requirements/torchserve.txt

# Java 설치 필요
sudo apt install openjdk-11-jdk
```

**서버 시작 (자동):**
```python
# 통합 인터페이스 사용 시 자동으로 서버가 시작/종료됨
from rm_abstract.serving import create_serving_engine, ServingConfig, ServingEngineType

with create_serving_engine(ServingConfig(engine=ServingEngineType.TORCHSERVE)) as engine:
    engine.load_model("gpt2")
    output = engine.infer("Hello")
```

**서버 시작 (수동):**
```bash
torchserve --start \
  --model-store ~/.rm_abstract/torchserve_models \
  --models all \
  --ncs \
  --disable-token-auth
```

### Rebellions NPU 설치

```bash
# RBLN SDK 설치 (하드웨어 필요)
pip install rebel-sdk

# 선택 1: vLLM-RBLN (고성능)
pip install vllm-rbln

# 선택 2: Optimum-RBLN (HuggingFace 통합)
pip install optimum-rbln
```

**참고:** https://docs.rbln.ai/latest/

---

## ✅ 설치 확인

### 시스템 검증

```bash
# 전체 검증 (실제 추론 테스트 포함)
python -m rm_abstract.system_validator

# 빠른 검증 (추론 테스트 제외)
python -m rm_abstract.system_validator --quick
```

### Python에서 확인

```python
import rm_abstract

# 시스템 정보 출력
rm_abstract.print_system_info()

# 검증 실행
rm_abstract.print_validation_report()

# 사용 가능한 백엔드 확인
backends = rm_abstract.get_available_backends()
print(backends)
```

### 테스트 실행

```bash
# pytest 테스트
pytest tests/test_core.py -v
pytest tests/test_api.py -v
```

---

## 🐛 문제 해결

### GPU 메모리 부족

```bash
# 다른 GPU 사용
CUDA_VISIBLE_DEVICES=1 python your_script.py

# 메모리 사용량 줄이기
export VLLM_GPU_MEMORY_UTILIZATION=0.5
```

### vLLM 멀티프로세싱 오류

```python
# 스크립트에 다음 추가
if __name__ == "__main__":
    # 코드를 여기에
    pass
```

### Triton 서버 연결 실패

```bash
# 서버 상태 확인
curl http://localhost:8000/v2/health/ready

# 로그 확인
docker logs rm_triton
```

### TorchServe Java 오류

```bash
# Java 버전 확인
java -version

# JAVA_HOME 설정
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
```

---

## 📁 디렉토리 구조

설치 후 생성되는 디렉토리:

```
~/.rm_abstract/
├── cache/              # 컴파일 캐시
├── torchserve_models/  # TorchServe 모델 저장소
└── logs/               # 로그 파일
```

---

## 🔗 관련 문서

- [QUICKSTART.md](QUICKSTART.md) - 빠른 시작 예제
- [ARCHITECTURE.md](ARCHITECTURE.md) - 시스템 아키텍처
- [API.md](API.md) - REST API 문서
