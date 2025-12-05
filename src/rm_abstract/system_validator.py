"""
System Validator Module

Actually tests whether each component works, not just checking if packages are installed.

Usage:
    python -m rm_abstract.system_validator
    
    # Or in Python
    from rm_abstract import validate_system, print_validation_report
    validate_system()
"""

import time
import logging
from typing import Dict, List, Optional, Any

from .validators.base import TestStatus, TestResult, ValidationReport

logger = logging.getLogger(__name__)


def _test_gpu_available() -> TestResult:
    """Test if GPU is actually accessible"""
    start = time.time()
    try:
        import torch
        
        if not torch.cuda.is_available():
            return TestResult(
                name="GPU Available",
                status=TestStatus.FAIL,
                message="CUDA not available",
                duration_ms=(time.time() - start) * 1000,
            )
        
        # Actually try to use GPU
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        
        # Simple tensor operation to verify GPU works
        x = torch.tensor([1.0, 2.0, 3.0]).cuda()
        y = x * 2
        result = y.cpu().numpy()
        
        return TestResult(
            name="GPU Available",
            status=TestStatus.PASS,
            message=f"{device_count} GPU(s): {device_name}",
            duration_ms=(time.time() - start) * 1000,
            details={"device_count": device_count, "device_name": device_name},
        )
    except ImportError:
        return TestResult(
            name="GPU Available",
            status=TestStatus.SKIP,
            message="PyTorch not installed",
            duration_ms=(time.time() - start) * 1000,
        )
    except Exception as e:
        return TestResult(
            name="GPU Available",
            status=TestStatus.FAIL,
            message=str(e),
            duration_ms=(time.time() - start) * 1000,
        )


def _find_free_gpu() -> Optional[int]:
    """Find a GPU with enough free memory"""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        
        best_gpu = None
        best_free_mem = 0
        
        for i in range(torch.cuda.device_count()):
            try:
                # Get memory info using nvidia-smi for accurate free memory
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", f"--id={i}"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    free_mem = int(result.stdout.strip())
                    if free_mem > best_free_mem:
                        best_free_mem = free_mem
                        best_gpu = i
            except:
                # Fallback: use torch to get memory info
                torch.cuda.set_device(i)
                free_mem = torch.cuda.mem_get_info(i)[0] / (1024**2)  # MB
                if free_mem > best_free_mem:
                    best_free_mem = free_mem
                    best_gpu = i
        
        # Need at least 2GB free for gpt2
        if best_free_mem > 2000:
            return best_gpu
        return None
    except:
        return None


def _test_vllm_gpu_inference() -> TestResult:
    """Test actual vLLM inference on GPU"""
    start = time.time()
    try:
        import torch
        if not torch.cuda.is_available():
            return TestResult(
                name="vLLM GPU Inference",
                status=TestStatus.SKIP,
                message="No GPU available",
                duration_ms=(time.time() - start) * 1000,
            )
        
        # Find GPU with enough free memory
        free_gpu = _find_free_gpu()
        if free_gpu is None:
            return TestResult(
                name="vLLM GPU Inference",
                status=TestStatus.WARN,
                message="All GPUs have insufficient memory (<2GB free)",
                duration_ms=(time.time() - start) * 1000,
            )
        
        from vllm import LLM, SamplingParams
        import os
        
        # Set to use the free GPU
        original_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(free_gpu)
        
        try:
            # Load a tiny model for testing with minimal memory
            llm = LLM(
                model="gpt2",
                trust_remote_code=True,
                gpu_memory_utilization=0.15,  # Use minimal memory for test
                max_model_len=128,
                enforce_eager=True,  # Disable CUDA graphs to save memory
            )
            
            # Run inference
            sampling_params = SamplingParams(max_tokens=10, temperature=0.0)
            outputs = llm.generate(["Hello"], sampling_params)
            
            generated_text = outputs[0].outputs[0].text
            
            # Cleanup
            del llm
            torch.cuda.empty_cache()
            
            return TestResult(
                name="vLLM GPU Inference",
                status=TestStatus.PASS,
                message=f"Generated: '{generated_text[:30]}...' (GPU:{free_gpu})",
                duration_ms=(time.time() - start) * 1000,
                details={"model": "gpt2", "output_length": len(generated_text), "gpu_id": free_gpu},
            )
        finally:
            # Restore original CUDA_VISIBLE_DEVICES
            if original_cuda_visible:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_visible
            else:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                
    except ImportError as e:
        return TestResult(
            name="vLLM GPU Inference",
            status=TestStatus.SKIP,
            message=f"vLLM not installed: {e}",
            duration_ms=(time.time() - start) * 1000,
        )
    except Exception as e:
        error_msg = str(e)
        if "memory" in error_msg.lower():
            return TestResult(
                name="vLLM GPU Inference",
                status=TestStatus.WARN,
                message="GPU memory insufficient - try freeing GPU memory",
                duration_ms=(time.time() - start) * 1000,
            )
        return TestResult(
            name="vLLM GPU Inference",
            status=TestStatus.FAIL,
            message=error_msg[:100],
            duration_ms=(time.time() - start) * 1000,
        )


def _test_cpu_inference() -> TestResult:
    """Test CPU inference with PyTorch"""
    start = time.time()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load model on CPU
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        model.eval()
        
        # Run inference
        inputs = tokenizer("Hello", return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return TestResult(
            name="CPU Inference",
            status=TestStatus.PASS,
            message=f"Generated: '{generated_text[:30]}...'",
            duration_ms=(time.time() - start) * 1000,
            details={"model": "gpt2", "output_length": len(generated_text)},
        )
    except ImportError as e:
        return TestResult(
            name="CPU Inference",
            status=TestStatus.SKIP,
            message=f"Required packages not installed: {e}",
            duration_ms=(time.time() - start) * 1000,
        )
    except Exception as e:
        return TestResult(
            name="CPU Inference",
            status=TestStatus.FAIL,
            message=str(e)[:100],
            duration_ms=(time.time() - start) * 1000,
        )


def _test_device_switching() -> TestResult:
    """Test GPU to CPU device switching"""
    start = time.time()
    try:
        import torch
        if not torch.cuda.is_available():
            return TestResult(
                name="Device Switching",
                status=TestStatus.SKIP,
                message="No GPU for switching test",
                duration_ms=(time.time() - start) * 1000,
            )
        
        import rm_abstract
        
        # Initialize with GPU
        rm_abstract.init(device="gpu:0", verbose=False)
        gpu_device = rm_abstract.get_controller().device_name
        
        # Switch to CPU
        rm_abstract.switch_device("cpu")
        cpu_device = rm_abstract.get_controller().device_name
        
        if "gpu" in gpu_device.lower() and "cpu" in cpu_device.lower():
            return TestResult(
                name="Device Switching",
                status=TestStatus.PASS,
                message=f"GPU → CPU switching works",
                duration_ms=(time.time() - start) * 1000,
                details={"from": gpu_device, "to": cpu_device},
            )
        else:
            return TestResult(
                name="Device Switching",
                status=TestStatus.WARN,
                message=f"Switch happened but devices unclear: {gpu_device} → {cpu_device}",
                duration_ms=(time.time() - start) * 1000,
            )
    except Exception as e:
        return TestResult(
            name="Device Switching",
            status=TestStatus.FAIL,
            message=str(e)[:100],
            duration_ms=(time.time() - start) * 1000,
        )


def _test_triton() -> TestResult:
    """Test Triton functionality (config generation + optional server check)"""
    start = time.time()
    try:
        import tritonclient.http as httpclient
        
        # First, test if we can create Triton model config
        try:
            from rm_abstract.serving import create_serving_engine, ServingConfig, ServingEngineType, DeviceTarget
            import tempfile
            import os
            
            with tempfile.TemporaryDirectory() as tmpdir:
                config = ServingConfig(
                    engine=ServingEngineType.TRITON,
                    device=DeviceTarget.GPU,
                    model_name="test_model",
                )
                engine = create_serving_engine(config)
                engine.setup_model_repository(tmpdir)
                engine.load_model("gpt2", triton_model_name="gpt2_test")
                
                # Check if config was created
                config_path = os.path.join(tmpdir, "gpt2_test", "config.pbtxt")
                if os.path.exists(config_path):
                    config_created = True
                else:
                    config_created = False
        except Exception as e:
            config_created = False
        
        # Then check if server is running (optional)
        server_running = False
        try:
            client = httpclient.InferenceServerClient(url="localhost:8000", verbose=False)
            server_running = client.is_server_live()
        except:
            pass
        
        if config_created and server_running:
            return TestResult(
                name="Triton",
                status=TestStatus.PASS,
                message="Config generation ✓, Server running ✓",
                duration_ms=(time.time() - start) * 1000,
                details={"config_created": True, "server_running": True},
            )
        elif config_created:
            return TestResult(
                name="Triton",
                status=TestStatus.PASS,
                message="Config generation ✓ (server not running)",
                duration_ms=(time.time() - start) * 1000,
                details={"config_created": True, "server_running": False},
            )
        else:
            return TestResult(
                name="Triton",
                status=TestStatus.WARN,
                message="Client installed, config generation failed",
                duration_ms=(time.time() - start) * 1000,
            )
            
    except ImportError:
        return TestResult(
            name="Triton",
            status=TestStatus.SKIP,
            message="tritonclient not installed",
            duration_ms=(time.time() - start) * 1000,
        )
    except Exception as e:
        return TestResult(
            name="Triton",
            status=TestStatus.FAIL,
            message=str(e)[:100],
            duration_ms=(time.time() - start) * 1000,
        )


def _test_torchserve() -> TestResult:
    """Test TorchServe functionality (.mar creation + optional server check)"""
    start = time.time()
    try:
        import subprocess
        
        # Check if torch-model-archiver is available
        result = subprocess.run(
            ["torch-model-archiver", "--help"],
            capture_output=True,
            timeout=5,
        )
        
        if result.returncode != 0:
            return TestResult(
                name="TorchServe",
                status=TestStatus.FAIL,
                message="torch-model-archiver not working",
                duration_ms=(time.time() - start) * 1000,
            )
        
        # Test .mar file creation
        mar_created = False
        try:
            from rm_abstract.serving import create_serving_engine, ServingConfig, ServingEngineType, DeviceTarget
            import tempfile
            import os
            
            with tempfile.TemporaryDirectory() as tmpdir:
                config = ServingConfig(
                    engine=ServingEngineType.TORCHSERVE,
                    device=DeviceTarget.GPU,
                    model_name="test_model",
                )
                engine = create_serving_engine(config)
                engine._model_store = tmpdir
                
                mar_path = engine.create_model_archive("gpt2", archive_name="gpt2_test")
                mar_created = mar_path and os.path.exists(mar_path)
        except Exception as e:
            mar_created = False
        
        # Check if server is running (optional)
        server_running = False
        try:
            import urllib.request
            response = urllib.request.urlopen("http://localhost:8080/ping", timeout=2)
            server_running = response.status == 200
        except:
            pass
        
        if mar_created and server_running:
            return TestResult(
                name="TorchServe",
                status=TestStatus.PASS,
                message=".mar creation ✓, Server running ✓",
                duration_ms=(time.time() - start) * 1000,
                details={"mar_created": True, "server_running": True},
            )
        elif mar_created:
            return TestResult(
                name="TorchServe",
                status=TestStatus.PASS,
                message=".mar creation ✓ (server not running, needs Java)",
                duration_ms=(time.time() - start) * 1000,
                details={"mar_created": True, "server_running": False},
            )
        else:
            return TestResult(
                name="TorchServe",
                status=TestStatus.WARN,
                message="torch-model-archiver ready, .mar creation failed",
                duration_ms=(time.time() - start) * 1000,
            )
            
    except FileNotFoundError:
        return TestResult(
            name="TorchServe",
            status=TestStatus.SKIP,
            message="torch-model-archiver not installed",
            duration_ms=(time.time() - start) * 1000,
        )
    except Exception as e:
        return TestResult(
            name="TorchServe",
            status=TestStatus.FAIL,
            message=str(e)[:100],
            duration_ms=(time.time() - start) * 1000,
        )


def _test_rbln_npu() -> TestResult:
    """Test Rebellions ATOM NPU"""
    start = time.time()
    try:
        # Try to import RBLN SDK
        try:
            import rebel
            sdk = "rebel"
        except ImportError:
            try:
                import rbln
                sdk = "rbln"
            except ImportError:
                return TestResult(
                    name="Rebellions NPU",
                    status=TestStatus.SKIP,
                    message="RBLN SDK not installed",
                    duration_ms=(time.time() - start) * 1000,
                )
        
        # Try to detect devices
        if sdk == "rebel":
            devices = rebel.get_devices()
        else:
            devices = rbln.get_devices()
        
        if len(devices) > 0:
            return TestResult(
                name="Rebellions NPU",
                status=TestStatus.PASS,
                message=f"{len(devices)} ATOM NPU(s) detected",
                duration_ms=(time.time() - start) * 1000,
                details={"device_count": len(devices), "sdk": sdk},
            )
        else:
            return TestResult(
                name="Rebellions NPU",
                status=TestStatus.WARN,
                message="SDK installed, no devices found",
                duration_ms=(time.time() - start) * 1000,
            )
    except Exception as e:
        return TestResult(
            name="Rebellions NPU",
            status=TestStatus.FAIL,
            message=str(e)[:100],
            duration_ms=(time.time() - start) * 1000,
        )


def validate_system(
    run_inference_tests: bool = True,
    verbose: bool = True,
) -> ValidationReport:
    """
    Run all system validation tests
    
    Args:
        run_inference_tests: Whether to run actual inference tests (slower but thorough)
        verbose: Print progress
        
    Returns:
        ValidationReport with all test results
    """
    from datetime import datetime
    
    report = ValidationReport(timestamp=datetime.now().isoformat())
    
    tests = [
        ("GPU Available", _test_gpu_available),
        ("CPU Inference", _test_cpu_inference),
    ]
    
    if run_inference_tests:
        tests.extend([
            ("vLLM GPU Inference", _test_vllm_gpu_inference),
            ("Device Switching", _test_device_switching),
        ])
    
    tests.extend([
        ("Triton", _test_triton),
        ("TorchServe", _test_torchserve),
        ("Rebellions NPU", _test_rbln_npu),
    ])
    
    for name, test_func in tests:
        if verbose:
            print(f"  Testing {name}...", end=" ", flush=True)
        
        result = test_func()
        report.add_result(result)
        
        if verbose:
            status_icon = {
                TestStatus.PASS: "✓",
                TestStatus.FAIL: "✗",
                TestStatus.SKIP: "○",
                TestStatus.WARN: "⚠",
            }[result.status]
            print(f"{status_icon} ({result.duration_ms:.0f}ms)")
    
    return report


def print_validation_report(report: Optional[ValidationReport] = None, run_inference_tests: bool = True) -> None:
    """
    Print formatted validation report
    
    Args:
        report: Existing report or None to generate new one
        run_inference_tests: Whether to run inference tests if generating new report
    """
    print()
    print("=" * 70)
    print("  RM Abstract Layer - System Validation")
    print("=" * 70)
    print()
    print("Running validation tests...")
    print()
    
    if report is None:
        report = validate_system(run_inference_tests=run_inference_tests, verbose=True)
    
    print()
    print("-" * 70)
    print("Results:")
    print("-" * 70)
    
    for result in report.results:
        status_icons = {
            TestStatus.PASS: "✅ PASS",
            TestStatus.FAIL: "❌ FAIL",
            TestStatus.SKIP: "⏭️  SKIP",
            TestStatus.WARN: "⚠️  WARN",
        }
        
        print(f"\n  {result.name}")
        print(f"    Status: {status_icons[result.status]}")
        print(f"    Message: {result.message}")
        if result.details:
            print(f"    Details: {result.details}")
    
    print()
    print("-" * 70)
    print("Summary:")
    print("-" * 70)
    print(f"  Total Tests: {report.total_tests}")
    print(f"  ✅ Passed:   {report.passed}")
    print(f"  ❌ Failed:   {report.failed}")
    print(f"  ⚠️  Warnings: {report.warnings}")
    print(f"  ⏭️  Skipped:  {report.skipped}")
    
    print()
    
    # Recommendations based on results
    print("💡 Recommendations:")
    
    gpu_result = next((r for r in report.results if r.name == "GPU Available"), None)
    vllm_result = next((r for r in report.results if r.name == "vLLM GPU Inference"), None)
    cpu_result = next((r for r in report.results if r.name == "CPU Inference"), None)
    
    if gpu_result and gpu_result.status == TestStatus.PASS:
        if vllm_result and vllm_result.status == TestStatus.PASS:
            print("  • GPU + vLLM ready for high-performance inference")
        elif vllm_result and vllm_result.status == TestStatus.FAIL:
            print("  • GPU available but vLLM failed - check GPU memory")
    
    if cpu_result and cpu_result.status == TestStatus.PASS:
        print("  • CPU inference available as fallback")
    
    triton_result = next((r for r in report.results if r.name == "Triton"), None)
    if triton_result and triton_result.status == TestStatus.PASS:
        if triton_result.details.get("server_running"):
            print("  • Triton server ready for multi-model serving")
        else:
            print("  • Triton config generation works - to start server:")
            print("      docker run --gpus=1 -p8000:8000 -v /models:/models nvcr.io/nvidia/tritonserver tritonserver")
    
    torchserve_result = next((r for r in report.results if r.name == "TorchServe"), None)
    if torchserve_result and torchserve_result.status == TestStatus.PASS:
        if torchserve_result.details.get("server_running"):
            print("  • TorchServe server ready for model serving")
        else:
            print("  • TorchServe .mar creation works - to start server (requires Java):")
            print("      torchserve --start --model-store ~/.rm_abstract/torchserve_models")
    
    rbln_result = next((r for r in report.results if r.name == "Rebellions NPU"), None)
    if rbln_result and rbln_result.status == TestStatus.SKIP:
        print("  • For Rebellions NPU: pip install rebel-sdk (requires hardware)")
    
    print()
    print("=" * 70)


def get_working_components() -> Dict[str, bool]:
    """
    Get a quick summary of what's actually working
    
    Returns:
        Dictionary of component name to working status
    """
    report = validate_system(run_inference_tests=False, verbose=False)
    
    return {
        result.name: result.status == TestStatus.PASS
        for result in report.results
    }


# CLI entry point
if __name__ == "__main__":
    import sys
    
    # Check for quick mode
    quick_mode = "--quick" in sys.argv or "-q" in sys.argv
    
    print_validation_report(run_inference_tests=not quick_mode)

