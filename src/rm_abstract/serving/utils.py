"""
Serving Engine Utilities

Common utilities for all serving engines.
"""

import logging
import time
import subprocess
import shutil
from typing import Optional, Callable, Any
from urllib.request import urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)


# ============================================================
# Server Health Check Utilities
# ============================================================

def wait_for_server(
    health_url: str,
    timeout: int = 60,
    check_interval: float = 1.0,
    success_status: int = 200,
) -> bool:
    """
    Wait for server to become ready.
    
    Args:
        health_url: URL to check for health (e.g., http://localhost:8000/health)
        timeout: Maximum time to wait in seconds
        check_interval: Time between checks in seconds
        success_status: HTTP status code indicating success
        
    Returns:
        True if server is ready, False if timeout
    """
    start = time.time()
    
    while time.time() - start < timeout:
        try:
            with urlopen(health_url, timeout=2) as response:
                if response.status == success_status:
                    logger.info(f"Server ready at {health_url}")
                    return True
        except (URLError, TimeoutError, ConnectionError):
            pass
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
        
        time.sleep(check_interval)
    
    logger.warning(f"Server not ready after {timeout}s: {health_url}")
    return False


def check_health(url: str, timeout: float = 2.0) -> bool:
    """
    Single health check.
    
    Args:
        url: Health check URL
        timeout: Request timeout
        
    Returns:
        True if healthy, False otherwise
    """
    try:
        with urlopen(url, timeout=timeout) as response:
            return bool(response.status == 200)
    except:
        return False


# ============================================================
# Process Management Utilities
# ============================================================

def find_executable(name: str) -> Optional[str]:
    """
    Find executable in PATH.
    
    Args:
        name: Executable name (e.g., 'docker', 'torchserve')
        
    Returns:
        Full path to executable or None
    """
    return shutil.which(name)


def run_command(
    cmd: list,
    timeout: Optional[int] = None,
    capture_output: bool = True,
) -> subprocess.CompletedProcess:
    """
    Run a command and return result.
    
    Args:
        cmd: Command and arguments as list
        timeout: Timeout in seconds
        capture_output: Whether to capture stdout/stderr
        
    Returns:
        CompletedProcess object
    """
    logger.debug(f"Running command: {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        capture_output=capture_output,
        text=True,
        timeout=timeout,
    )


def start_background_process(
    cmd: list,
    cwd: Optional[str] = None,
) -> subprocess.Popen:
    """
    Start a background process.
    
    Args:
        cmd: Command and arguments
        cwd: Working directory
        
    Returns:
        Popen object
    """
    logger.info(f"Starting background process: {' '.join(cmd)}")
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
    )


def stop_process(process: subprocess.Popen, timeout: int = 10) -> None:
    """
    Stop a process gracefully.
    
    Args:
        process: Process to stop
        timeout: Time to wait for graceful shutdown before killing
    """
    if process is None:
        return
    
    try:
        process.terminate()
        process.wait(timeout=timeout)
        logger.info("Process terminated gracefully")
    except subprocess.TimeoutExpired:
        logger.warning("Process did not terminate, killing...")
        process.kill()
        process.wait()


# ============================================================
# Docker Utilities
# ============================================================

def is_docker_available() -> bool:
    """Check if Docker is available and accessible."""
    docker = find_executable("docker")
    if docker is None:
        return False
    
    try:
        result = run_command(["docker", "ps"], timeout=5)
        return result.returncode == 0
    except:
        return False


def docker_container_exists(name: str) -> bool:
    """Check if a Docker container exists."""
    try:
        result = run_command(
            ["docker", "ps", "-a", "-q", "-f", f"name={name}"],
            timeout=5,
        )
        return bool(result.stdout.strip())
    except:
        return False


def docker_stop_container(name: str, timeout: int = 10) -> bool:
    """
    Stop and remove a Docker container.
    
    Args:
        name: Container name
        timeout: Stop timeout
        
    Returns:
        True if successful
    """
    try:
        run_command(["docker", "stop", "-t", str(timeout), name], timeout=timeout + 5)
        run_command(["docker", "rm", "-f", name], timeout=5)
        return True
    except:
        return False


def docker_run(
    image: str,
    name: str,
    ports: Optional[dict] = None,
    volumes: Optional[dict] = None,
    gpus: Optional[str] = None,
    command: Optional[list] = None,
    detach: bool = True,
    remove: bool = True,
) -> Optional[str]:
    """
    Run a Docker container.
    
    Args:
        image: Docker image name
        name: Container name
        ports: Port mappings {host: container}
        volumes: Volume mappings {host: container}
        gpus: GPU specification (e.g., "all", "0,1")
        command: Command to run
        detach: Run in background
        remove: Remove container when stopped
        
    Returns:
        Container ID or None on failure
    """
    cmd = ["docker", "run"]
    
    if detach:
        cmd.append("-d")
    if remove:
        cmd.append("--rm")
    
    cmd.extend(["--name", name])
    
    if gpus:
        cmd.extend(["--gpus", gpus])
    
    if ports:
        for host_port, container_port in ports.items():
            cmd.extend(["-p", f"{host_port}:{container_port}"])
    
    if volumes:
        for host_path, container_path in volumes.items():
            cmd.extend(["-v", f"{host_path}:{container_path}"])
    
    cmd.append(image)
    
    if command:
        cmd.extend(command)
    
    logger.info(f"Docker run: {' '.join(cmd)}")
    
    try:
        result = run_command(cmd, timeout=60)
        if result.returncode == 0:
            container_id: str = result.stdout.strip()
            logger.info(f"Container started: {container_id[:12]}")
            return container_id
        else:
            logger.error(f"Docker run failed: {result.stderr}")
            return None
    except Exception as e:
        logger.error(f"Docker run error: {e}")
        return None


# ============================================================
# Model Utilities
# ============================================================

def get_model_name(model_name_or_path: str) -> str:
    """
    Extract clean model name from path or HuggingFace name.
    
    Args:
        model_name_or_path: Full model path or name
        
    Returns:
        Clean model name suitable for file/directory names
    """
    # Remove common prefixes
    name = model_name_or_path
    if "/" in name:
        name = name.split("/")[-1]
    
    # Remove file extensions
    for ext in [".bin", ".safetensors", ".pt", ".onnx"]:
        if name.endswith(ext):
            name = name[:-len(ext)]
    
    return name.replace("-", "_").replace(".", "_").lower()

