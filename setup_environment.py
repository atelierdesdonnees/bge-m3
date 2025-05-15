"""
Environment setup script for VLLM deployment.

This script handles the setup of both Docker and bare server environments for VLLM,
including system packages, Python dependencies, directory structure, and file management.
"""

import os
import sys
import logging
import subprocess
import argparse
import shutil
from pathlib import Path
from typing import Dict, Optional, Union, NoReturn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # This ensures output to stdout
)

HERE = Path(__file__).parent


def run_command(command: str, check: bool = True) -> Optional[NoReturn]:
    """
    Execute a shell command and handle potential errors.

    Args:
        command: Shell command to execute
        check: If True, raises CalledProcessError on non-zero exit status

    Raises:
        subprocess.CalledProcessError: If the command fails and check is True
        SystemExit: If check is True and command fails
    """
    try:
        subprocess.run(command, check=check, shell=True)
    except subprocess.CalledProcessError as e:
        logging.info(f"Error executing command: {command}")
        logging.info(f"Error: {str(e)}")
        if check:
            sys.exit(1)


# def setup_basic_environment() -> None:
#     """
#     Setup basic system environment including apt packages.

#     Performs system updates, installs required packages, and cleans up
#     unnecessary files to minimize environment size.

#     System packages installed:
#     - python3 and development tools
#     - build essentials
#     """
#     run_command("apt-get update -y")
#     run_command("apt-get upgrade -y")

#     packages: List[str] = [
#         "python3-pip",
#         "python3-minimal",
#         "python3-setuptools",
#         "python3-dev",
#         "build-essential",
#         "git",
#         "poppler-utils",
#         "libmagic1",
#     ]
#     run_command(f"apt-get install -y --no-install-recommends {' '.join(packages)}")
#     # Cleanup after apt operations
#     run_command("apt-get clean")
#     run_command("rm -rf /var/lib/apt/lists/*")
#     run_command("rm -rf /tmp/* /var/tmp/*")


def setup_cuda(cuda_version: str) -> None:
    """
    Configure CUDA environment for the specified version.

    Args:
        cuda_version: Version string of CUDA to configure
    """
    run_command(f"ldconfig /usr/local/{cuda_version}/compat/")


def install_python_packages(worker_infinity_requirements: Union[str, Path], additional_requirements: Optional[Union[str, Path]] = None) -> None:
    """
    Install required Python packages from multiple requirement files.

    Args:
        worker_infinity_requirements: Path to worker-infinity's requirements.txt
        additional_requirements: Optional path to additional project requirements
    """
    # First upgrade pip alone
    run_command("python3 -m pip install --no-cache-dir --upgrade pip")

    # Install worker-infinity requirements
    run_command(f"python3 -m pip install --no-cache-dir --upgrade -r {worker_infinity_requirements}")

    # Install additional requirements if provided
    if additional_requirements and (_req := HERE / additional_requirements).exists():
        run_command(f"python3 -m pip install --no-cache-dir -r {_req}")
    else:
        logging.info("Le fichier des requirements personnalisé n'a pas été trouvé.")


def setup_infinity_environment(src_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Setup VLLM environment and override source files.

    Clones the worker-infinity repository into the parent directory and optionally
    overrides its source files with custom implementations.

    Args:
        src_dir: Optional directory containing custom source files to override
                worker-infinity implementations

    Returns:
        Path: Path to the worker-infinity directory

    Note:
        worker-infinity is cloned in the parent directory to avoid conflicts
        with the current git repository
    """
    # Ensure we use absolute paths
    worker_infinity_path = Path(__file__).parent.parent / "worker-infinity-embedding"

    # Always ensure a fresh clone of worker-infinity in parent directory
    if worker_infinity_path.exists():
        shutil.rmtree(worker_infinity_path)

    run_command(f"git clone https://github.com/runpod-workers/worker-infinity-embedding.git {worker_infinity_path}")

    # Override source files if src_dir is provided
    if src_dir and (path_src_dir := (Path(__file__).parent / src_dir)).exists():
        vllm_src_dir = worker_infinity_path / "src"
        vllm_src_dir.mkdir(parents=True, exist_ok=True)
        for py_file in path_src_dir.glob("*.py"):
            shutil.copy(str(py_file), str(vllm_src_dir))

    return worker_infinity_path


def setup_base_directories() -> str:
    """
    Setup base directories for models and templates.

    Creates the necessary directory structure for models, caches, and templates.
    Uses BASE_PATH from environment variables or defaults to '/models'.

    Returns:
        str: The base path used for the directory structure
    """
    base_path = os.environ.get("BASE_PATH", "/models")

    directories = [
        base_path,
        f"{base_path}/huggingface-cache/datasets",
        f"{base_path}/huggingface-cache/hub",
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def load_env_file(env_path: Union[str, Path]) -> Dict[str, str]:
    """
    Load environment variables from file.

    Reads environment variables from a file and sets them in the current
    environment. Handles variable expansion (e.g., ${VAR}).

    Args:
        env_path: Path to environment file

    Returns:
        Dict[str, str]: Dictionary of loaded environment variables

    Note:
        Variables defined in the file override existing environment variables
    """
    if not Path(env_path).exists():
        logging.info(f"Warning: Environment file {env_path} not found")
        return {}

    env_vars: Dict[str, str] = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                key, value = line.split("=", 1)
                # Handle variable expansion
                value = os.path.expandvars(value)
                os.environ[key] = value
                env_vars[key] = value
    return env_vars


def copy_files(env_file: Union[str, Path], src_dir: Union[str, Path]) -> None:
    """
    Copy configuration and source files to their deployment locations.

    Args:
        env_file: Path to environment.env file
        src_dir: Directory containing source files
        worker_infinity_path: Path to worker-vllm directory
    """
    if isinstance(env_file, str):
        env_file = HERE / env_file
    if isinstance(src_dir, str):
        src_dir = HERE / src_dir

    # Create /root directory if it doesn't exist (for Podman compatibility)
    Path("/root").mkdir(parents=True, exist_ok=True)

    # Always ensure environment.env is in /root if provided
    if env_file and Path(env_file).exists():
        print(f"Copying {env_file} to /root/.env")
        shutil.copy(env_file, "/root/.env")

    # Setup base directories and get BASE_PATH
    setup_base_directories()


def _download_model(src_folder: Path = (Path(__file__).parent / "worker-vllm" / "src")) -> None:
    sys.path.append(str(src_folder))
    from download_model import __call__

    __call__()


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup environment for VLLM")
    parser.add_argument("--cuda-version", help="CUDA version (overrides env file)")
    parser.add_argument("--src-dir", help="Directory containing source files", default="./")
    parser.add_argument("--env-file", help="Path to environment.env file", default="environment.env")
    parser.add_argument("--requirements-file", help="Path to additional requirements.txt file", default="requirements.txt")

    args = parser.parse_args()

    if args.env_file:
        load_env_file(args.env_file)

    worker_infinity_path = setup_infinity_environment(args.src_dir)

    copy_files(env_file=args.env_file, src_dir=args.src_dir)

    # setup_basic_environment()
    setup_cuda(args.cuda_version)

    # Install both sets of requirements
    install_python_packages(
        worker_infinity_requirements=f"{worker_infinity_path}/requirements.txt",
        additional_requirements=args.requirements_file,
    )

    _download_model(src_folder=worker_infinity_path / "src")


if __name__ == "__main__":
    main()
