# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #
import io
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install

try:
    torch_available = True
    # The assert is not needed since Github CI does not use GPU server,
    # install cuda library is sufficient
    # assert torch.cuda.is_available() == True
    from torch.utils.cpp_extension import CUDA_HOME
    from torch.utils.cpp_extension import ROCM_HOME
except Exception:
    torch_available = False
    CUDA_HOME = None
    ROCM_HOME = None
    print(
        "[WARNING] Unable to import torch, pre-compiling ops will be disabled. "
        "Please visit https://pytorch.org/ to see how to properly install torch on your system."  # noqa: E501
    )


ROOT_DIR = os.path.dirname(__file__)

# Determine build backend based on environment and platform
def detect_build_backend():
    """Detect which backend to build based on environment variables and platform."""
    
    # Check for explicit backend selection
    use_cann = os.getenv("USE_CANN", "0").lower() in ("1", "true", "yes")
    use_cuda = os.getenv("USE_CUDA", "auto").lower()
    
    # CANN support (Linux only)
    ascend_toolkit_home = os.getenv("ASCEND_TOOLKIT_HOME")
    # ! verify if the platform.system() is correct.
    if use_cann and ascend_toolkit_home and (platform.system() == "Linux" or platform.system() == "OpenEuler"):
        if check_cann_available(ascend_toolkit_home):
            return "cann", ascend_toolkit_home
        else:
            print("[WARNING] CANN requested but not available, falling back...")
    
    # CUDA support
    if use_cuda != "false" and CUDA_HOME is not None:
        if check_nvcc_installed(CUDA_HOME):
            return "cuda", CUDA_HOME
        else:
            print("[WARNING] CUDA requested but nvcc not available, falling back...")
    
    # ROCm support
    if ROCM_HOME is not None:
        if check_hipcc_installed(ROCM_HOME):
            return "rocm", ROCM_HOME
        else:
            print("[WARNING] ROCm available but hipcc not working, falling back...")
    
    # CPU-only fallback
    print("[INFO] Building in CPU-only mode")
    return "cpu", None


def check_cann_available(ascend_toolkit_home: str) -> bool:
    """Check if CANN toolkit is available."""
    try:
        # Check for essential CANN headers
        include_dir = os.path.join(ascend_toolkit_home, "include")
        acl_header = os.path.join(include_dir, "acl", "acl.h")
        
        if not os.path.exists(acl_header):
            print(f"[WARNING] CANN header not found: {acl_header}")
            return False
            
        # Check for CANN libraries
        lib_dir = os.path.join(ascend_toolkit_home, "lib64")
        acl_lib = os.path.join(lib_dir, "libascendcl.so")
        
        if not os.path.exists(acl_lib):
            print(f"[WARNING] CANN library not found: {acl_lib}")
            return False
            
        print(f"[INFO] CANN toolkit found at: {ascend_toolkit_home}")
        return True
    except Exception as e:
        print(f"[WARNING] Error checking CANN availability: {e}")
        return False


def check_nvcc_installed(cuda_home: str) -> bool:
    """Check if nvcc (NVIDIA CUDA compiler) is installed."""
    try:
        _ = subprocess.check_output(
            [cuda_home + "/bin/nvcc", "-V"], universal_newlines=True
        )
        return True
    except Exception:
        print("[WARNING] nvcc is not installed or not found in your PATH.")
        return False


def check_hipcc_installed(rocm_home: str) -> bool:
    """Check if hipcc (AMD HIP compiler) is installed."""
    # can be either <ROCM_HOME>/hip/bin/hipcc or <ROCM_HOME>/bin/hipcc
    hipcc_paths = [rocm_home + "/bin/hipcc", rocm_home + "/hip/bin/hipcc"]
    for hipcc in hipcc_paths:
        try:
            _ = subprocess.check_output(
                [hipcc, "--version"], universal_newlines=True
            )
            return True
        except Exception:
            continue
    print("[WARNING] hipcc is not installed or not found in your PATH.")
    return False


# Detect build backend
BUILD_BACKEND, BACKEND_HOME = detect_build_backend()
print(f"[INFO] Building with backend: {BUILD_BACKEND}")

def is_ninja_available() -> bool:
    try:
        subprocess.run(["ninja", "--version"], stdout=subprocess.PIPE)
    except FileNotFoundError:
        return False
    return True


class CustomInstall(install):
    """Custom installation to ensure proto files are compiled
    and extensions are built before installation."""

    def run(self):
        self.run_command("build_ext")
        super().run()


def fetch_requirements(path):
    with open(path, "r") as fd:
        return [r.strip() for r in fd.readlines()]


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def read_readme() -> str:
    """Read the README file if present."""
    p = get_path("README.md")
    if os.path.isfile(p):
        return io.open(get_path("README.md"), "r", encoding="utf-8").read()
    else:
        return ""


install_requires = fetch_requirements("requirements.txt")

extras = {}

extras["test"] = [
    "pytest",
    "accelerate>=0.27.2",
    "transformers>=4.37.2",
    "parameterized",
]

sys.path.append(Path.cwd().as_posix())


class CMakeExtension(Extension):
    def __init__(self, name: str, cmake_lists_dir: str = ".", **kwa) -> None:
        super().__init__(name, sources=[], **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


# Adapted from https://github.com/vllm-project/vllm/blob/a1242324c99ff8b1e29981006dfb504da198c7c3/setup.py
# Pass -DUSE_CANN=ON to the cmake command to build the extension with CANN support.
class cmake_build_ext(build_ext):
    did_config: Dict[str, bool] = {}

    def configure(self, ext: CMakeExtension) -> None:
        if ext.cmake_lists_dir in cmake_build_ext.did_config:
            return

        cmake_build_ext.did_config[ext.cmake_lists_dir] = True

        default_cfg = "Debug" if self.debug else "Release"
        cfg = os.getenv("CMAKE_BUILD_TYPE", default_cfg)

        outdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name))
        )

        cmake_args = [
            "-DCMAKE_BUILD_TYPE={}".format(cfg),
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(outdir),
            "-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={}".format(outdir),
            "-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={}".format(self.build_temp),
        ]

        # Add backend-specific arguments
        if BUILD_BACKEND == "cann":
            cmake_args.extend([
                "-DUSE_CANN=ON",
                "-DASCEND_TOOLKIT_HOME={}".format(BACKEND_HOME),
            ])
        elif BUILD_BACKEND == "cuda":
            cmake_args.extend([
                "-DUSE_CUDA=ON",
                "-DCUDA_TOOLKIT_ROOT_DIR={}".format(BACKEND_HOME),
            ])
        elif BUILD_BACKEND == "rocm":
            cmake_args.extend([
                "-DUSE_ROCM=ON",
                "-DROCM_ROOT={}".format(BACKEND_HOME),
            ])
        else:  # cpu
            cmake_args.extend([
                "-DUSE_CPU_ONLY=ON",
            ])

        # verbose = bool(int(os.getenv('VERBOSE', '1')))
        verbose = True
        if verbose:
            cmake_args += ["-DCMAKE_VERBOSE_MAKEFILE=ON"]

        cmake_args += [
            "-DSLLM_STORE_PYTHON_EXECUTABLE={}".format(sys.executable)
        ]

        if is_ninja_available():
            build_tool = ["-G", "Ninja"]
            cmake_args += [
                "-DCMAKE_JOB_POOL_COMPILE:STRING=compile",
                "-DCMAKE_JOB_POOLS:STRING=compile={}".format(8),
            ]
        else:
            # Default build tool to whatever cmake picks.
            build_tool = []

        try:
            subprocess.check_call(
                ["cmake", ext.cmake_lists_dir, *build_tool, *cmake_args],
                cwd=self.build_temp,
            )
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] CMake configuration failed with backend {BUILD_BACKEND}")
            print(f"[ERROR] Command: cmake {ext.cmake_lists_dir} {' '.join(build_tool)} {' '.join(cmake_args)}")
            raise e

    def build_extensions(self) -> None:
        # Ensure that CMake is present and working
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as e:
            raise RuntimeError("Cannot find CMake executable") from e

        # Create build directory if it does not exist.
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Build all the extensions
        for ext in self.extensions:
            self.configure(ext)

            ext_target_name = remove_prefix(ext.name, "sllm_store.")
            # num_jobs = 32

            build_args = [
                "--build",
                ".",
                "--target",
                ext_target_name,
                "-j",
                # str(num_jobs)
            ]

            try:
                subprocess.check_call(["cmake", *build_args], cwd=self.build_temp)
                print(f"[INFO] Successfully built extension: {ext_target_name}")
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Failed to build extension: {ext_target_name}")
                raise e
            print(self.build_temp, ext_target_name)


cmdclass = {
    "build_ext": cmake_build_ext,
    "install": CustomInstall,
}

setup(
    name="serverless-llm-store",
    version="0.6.3",
    ext_modules=[
        CMakeExtension(name="sllm_store._C"),
        CMakeExtension(name="sllm_store._checkpoint_store"),
    ],
    install_requires=install_requires,
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    extras_require=extras,
    #  cli entry point
    entry_points={
        "console_scripts": ["sllm-store=sllm_store.cli:main"],
    },
    package_data={
        "sllm_store": ["py.typed", "*.so"],
    },
    packages=[
        "sllm_store",
        "sllm_store.proto",
    ],
    cmdclass=cmdclass,
)
