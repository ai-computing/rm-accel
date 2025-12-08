"""
Migration Guide: Native vLLM → RM Abstract Layer

This file shows the minimal changes needed to migrate from native vLLM
to RM Abstract Layer, enabling future device switching (GPU→CPU→NPU).

Comparison:
    - Native vLLM: GPU only, no flexibility
    - RM Abstract: Same code can run on GPU, CPU, NPU, Triton, TorchServe

Migration effort: ~3-5 lines of code change
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================================================
#  BEFORE: Native vLLM Script (GPU Only)
# ============================================================

def native_vllm_example():
    """
    기존 vLLM 스크립트 - GPU에서만 동작
    
    이 코드는 vLLM만 사용하므로:
    - GPU 없으면 실행 불가
    - CPU나 NPU로 전환하려면 전체 코드 재작성 필요
    """
    from vllm import LLM, SamplingParams
    
    # 모델 로드 (GPU 고정)
    llm = LLM(model="gpt2")
    
    # 추론
    prompts = ["Hello, I am", "The future of AI is"]
    sampling_params = SamplingParams(temperature=0.7, max_tokens=50)
    
    outputs = llm.generate(prompts, sampling_params)
    
    for output in outputs:
        print(f"Prompt: {output.prompt}")
        print(f"Generated: {output.outputs[0].text}")
        print()


# ============================================================
#  AFTER: RM Abstract Layer (Flexible)
# ============================================================

def rm_abstract_example():
    """
    RM Abstract Layer 사용 - GPU, CPU, NPU 모두 지원
    
    변경점:
    1. import rm_abstract 추가 (1줄)
    2. rm_abstract.init() 호출 (1줄)  
    3. 나머지 코드는 동일!
    
    장점:
    - 나중에 CPU로 전환: rm_abstract.init(device="cpu")
    - 나중에 NPU로 전환: rm_abstract.init(device="npu")
    - 코드 수정 없이 환경 변수로도 가능: RM_DEVICE=cpu
    """
    import rm_abstract                    # ← 추가 (1줄)
    from vllm import LLM, SamplingParams
    
    rm_abstract.init(device="gpu")        # ← 추가 (1줄)
    
    # 이하 동일!
    llm = LLM(model="gpt2")
    
    prompts = ["Hello, I am", "The future of AI is"]
    sampling_params = SamplingParams(temperature=0.7, max_tokens=50)
    
    outputs = llm.generate(prompts, sampling_params)
    
    for output in outputs:
        print(f"Prompt: {output.prompt}")
        print(f"Generated: {output.outputs[0].text}")
        print()


# ============================================================
#  코드 비교 (Side by Side)
# ============================================================

BEFORE_CODE = '''
from vllm import LLM, SamplingParams

llm = LLM(model="gpt2")
prompts = ["Hello, I am"]
params = SamplingParams(temperature=0.7, max_tokens=50)
outputs = llm.generate(prompts, params)
'''

AFTER_CODE = '''
import rm_abstract                      # ← +1줄
from vllm import LLM, SamplingParams

rm_abstract.init(device="gpu")          # ← +1줄

llm = LLM(model="gpt2")
prompts = ["Hello, I am"]
params = SamplingParams(temperature=0.7, max_tokens=50)
outputs = llm.generate(prompts, params)
'''


# ============================================================
#  더 유연한 방식: UnifiedModel API
# ============================================================

def unified_model_example():
    """
    UnifiedModel 사용 - 엔진 변경도 2줄로!
    
    이 방식을 사용하면:
    - vLLM → Triton: engine="triton"만 변경
    - GPU → CPU: device="cpu"만 변경
    """
    from rm_abstract import UnifiedModel
    
    # vLLM GPU
    model = UnifiedModel("gpt2", engine="vllm", device="gpu")
    
    # 같은 generate() API
    output = model.generate("Hello, I am", max_tokens=50)
    print(f"Output: {output}")
    
    model.close()


UNIFIED_VLLM_GPU = '''
from rm_abstract import UnifiedModel

model = UnifiedModel("gpt2", engine="vllm", device="gpu")
output = model.generate("Hello", max_tokens=50)
'''

UNIFIED_TRITON_CPU = '''
from rm_abstract import UnifiedModel

model = UnifiedModel("gpt2", engine="triton", device="cpu")  # ← 이것만 변경!
output = model.generate("Hello", max_tokens=50)
'''


# ============================================================
#  MAIN
# ============================================================

def show_comparison():
    print("=" * 70)
    print("  Migration Guide: Native vLLM → RM Abstract Layer")
    print("=" * 70)
    
    print("\n" + "-" * 70)
    print("  BEFORE (Native vLLM) - GPU Only")
    print("-" * 70)
    print(BEFORE_CODE)
    
    print("-" * 70)
    print("  AFTER (RM Abstract) - GPU/CPU/NPU 모두 지원")
    print("-" * 70)
    print(AFTER_CODE)
    
    print("-" * 70)
    print("  변경 요약")
    print("-" * 70)
    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │  변경 사항                                                   │
    ├─────────────────────────────────────────────────────────────┤
    │  + import rm_abstract              (1줄 추가)               │
    │  + rm_abstract.init(device="gpu")  (1줄 추가)               │
    │                                                             │
    │  나머지 코드: 변경 없음!                                      │
    ├─────────────────────────────────────────────────────────────┤
    │  총 수정량: 2줄                                              │
    └─────────────────────────────────────────────────────────────┘
    """)
    
    print("-" * 70)
    print("  UnifiedModel API (더 유연한 방식)")
    print("-" * 70)
    print("\n  vLLM GPU:")
    print(UNIFIED_VLLM_GPU)
    print("  Triton CPU (같은 generate() API!):")
    print(UNIFIED_TRITON_CPU)
    
    print("-" * 70)
    print("  Migration Benefits")
    print("-" * 70)
    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │  Native vLLM           │  RM Abstract Layer                 │
    ├────────────────────────┼────────────────────────────────────┤
    │  GPU only              │  GPU, CPU, NPU 모두 지원            │
    │  vLLM only             │  vLLM, Triton, TorchServe 지원      │
    │  하드코딩된 디바이스      │  런타임 디바이스 전환 가능           │
    │  전환 시 전체 재작성     │  전환 시 1-2줄만 변경               │
    └────────────────────────┴────────────────────────────────────┘
    """)


def run_actual_test():
    """실제 테스트 (HuggingFace로 - vLLM 없어도 동작)"""
    print("\n" + "=" * 70)
    print("  실제 테스트 (HuggingFace backend)")
    print("=" * 70)
    
    import rm_abstract
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("\n  1. rm_abstract 초기화...")
    rm_abstract.init(device="cpu", verbose=False)
    
    print("  2. 모델 로드...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    print("  3. 추론 실행...")
    inputs = tokenizer("Hello, I am", return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=20, do_sample=True)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"\n  [Input]  \"Hello, I am\"")
    print(f"  [Output] \"{text}\"")
    print("\n  ✓ 테스트 완료!")


if __name__ == "__main__":
    show_comparison()
    
    # 실제 테스트 실행
    try:
        run_actual_test()
    except Exception as e:
        print(f"\n  테스트 스킵 (옵션): {e}")
    
    print("\n" + "=" * 70)
    print("  Migration Guide Complete!")
    print("=" * 70)

