# Copyright (c) 2020 Jisang Yoon
# All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/rmcgibbo/npcuda-example and
# https://github.com/cupy/cupy/blob/master/cupy_setup_build.py
# pylint: disable=fixme,access-member-before-definition
# pylint: disable=attribute-defined-outside-init,arguments-differ
import logging
import os
import sys
import subprocess
import re
import ctypes
import json
from typing import Any, Dict, List
from warnings import warn

from distutils import ccompiler, errors, msvccompiler, unixccompiler
from setuptools.command.build_ext import build_ext as setuptools_build_ext


HALF_PRECISION = False


# One of the following libraries must be available to load
libnames = ('libcuda.so', 'libcuda.dylib', 'cuda.dll')
for libname in libnames:
    try:
        cuda = ctypes.CDLL(libname)
    except OSError:
        continue
    else:
        break
else:
    raise ImportError(f'Could not load any of: {", ".join(libnames)}')

# Constants from cuda.h
CUDA_SUCCESS = 0
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39
CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36

# Conversions from semantic version numbers
# Borrowed from original gist and updated from the "GPUs supported" section of this Wikipedia article
# https://en.wikipedia.org/wiki/CUDA
SEMVER_TO_CORES = {
    (1, 0): 8,    # Tesla
    (1, 1): 8,
    (1, 2): 8,
    (1, 3): 8,
    (2, 0): 32,   # Fermi
    (2, 1): 48,
    (3, 0): 192,  # Kepler
    (3, 2): 192,
    (3, 5): 192,
    (3, 7): 192,
    (5, 0): 128,  # Maxwell
    (5, 2): 128,
    (5, 3): 128,
    (6, 0): 64,   # Pascal
    (6, 1): 128,
    (6, 2): 128,
    (7, 0): 64,   # Volta
    (7, 2): 64,
    (7, 5): 64,   # Turing
    (8, 0): 64,   # Ampere
    (8, 6): 64,
}
SEMVER_TO_ARCH = {
    (1, 0): 'tesla',
    (1, 1): 'tesla',
    (1, 2): 'tesla',
    (1, 3): 'tesla',

    (2, 0): 'fermi',
    (2, 1): 'fermi',

    (3, 0): 'kepler',
    (3, 2): 'kepler',
    (3, 5): 'kepler',
    (3, 7): 'kepler',

    (5, 0): 'maxwell',
    (5, 2): 'maxwell',
    (5, 3): 'maxwell',

    (6, 0): 'pascal',
    (6, 1): 'pascal',
    (6, 2): 'pascal',

    (7, 0): 'volta',
    (7, 2): 'volta',

    (7, 5): 'turing',

    (8, 0): 'ampere',
    (8, 6): 'ampere',
}


def get_cuda_device_specs() -> List[Dict[str, Any]]:
    """Generate spec for each GPU device with format

    {
        'name': str,
        'compute_capability': (major: int, minor: int),
        'cores': int,
        'cuda_cores': int,
        'concurrent_threads': int,
        'gpu_clock_mhz': float,
        'mem_clock_mhz': float,
        'total_mem_mb': float,
        'free_mem_mb': float
    }
    """

    # Type-binding definitions for ctypes
    num_gpus = ctypes.c_int()
    name = b' ' * 100
    cc_major = ctypes.c_int()
    cc_minor = ctypes.c_int()
    cores = ctypes.c_int()
    threads_per_core = ctypes.c_int()
    clockrate = ctypes.c_int()
    free_mem = ctypes.c_size_t()
    total_mem = ctypes.c_size_t()
    result = ctypes.c_int()
    device = ctypes.c_int()
    context = ctypes.c_void_p()
    error_str = ctypes.c_char_p()

    # Check expected initialization
    result = cuda.cuInit(0)
    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        raise RuntimeError(f'cuInit failed with error code {result}: {error_str.value.decode()}')
    result = cuda.cuDeviceGetCount(ctypes.byref(num_gpus))
    if result != CUDA_SUCCESS:
        cuda.cuGetErrorString(result, ctypes.byref(error_str))
        raise RuntimeError(f'cuDeviceGetCount failed with error code {result}: {error_str.value.decode()}')

    # Iterate through available devices
    device_specs = []
    for i in range(num_gpus.value):
        spec = {}
        result = cuda.cuDeviceGet(ctypes.byref(device), i)
        if result != CUDA_SUCCESS:
            cuda.cuGetErrorString(result, ctypes.byref(error_str))
            raise RuntimeError(f'cuDeviceGet failed with error code {result}: {error_str.value.decode()}')

        # Parse specs for each device
        if cuda.cuDeviceGetName(ctypes.c_char_p(name), len(name), device) == CUDA_SUCCESS:
            spec.update(name=name.split(b'\0', 1)[0].decode())
        if cuda.cuDeviceComputeCapability(ctypes.byref(cc_major), ctypes.byref(cc_minor), device) == CUDA_SUCCESS:
            spec.update(compute_capability=(cc_major.value, cc_minor.value))
            spec.update(architecture=SEMVER_TO_ARCH.get((cc_major.value, cc_minor.value), 'unknown'))
        if cuda.cuDeviceGetAttribute(ctypes.byref(cores), CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device) == CUDA_SUCCESS:
            spec.update(
                cores=cores.value,
                cuda_cores=cores.value * SEMVER_TO_CORES.get((cc_major.value, cc_minor.value), 'unknown'))
            if cuda.cuDeviceGetAttribute(ctypes.byref(threads_per_core), CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device) == CUDA_SUCCESS:
                spec.update(concurrent_threads=cores.value * threads_per_core.value)
        if cuda.cuDeviceGetAttribute(ctypes.byref(clockrate), CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device) == CUDA_SUCCESS:
            spec.update(gpu_clock_mhz=clockrate.value / 1000.)
        if cuda.cuDeviceGetAttribute(ctypes.byref(clockrate), CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device) == CUDA_SUCCESS:
            spec.update(mem_clock_mhz=clockrate.value / 1000.)
        device_specs.append(spec)
    return device_specs
  

def prepare_cuda():
    """Locate the CUDA environment on the system
    If a valid CUDA installation is found
    this returns a dictionary with keys 'home', 'nvcc', 'include',
    and 'lib64' and values giving the absolute path to each directory.
    """
    cudaconfig = {}

    # Check if CUDA is installed
    try:
        output = subprocess.check_output(['nvcc', '--version'], stderr=subprocess.STDOUT)
    except OSError:
        logging.warning('nvcc not found. CUDA is not installed')
        return None

    # Extract CUDA version from output
    version_str = output.decode('utf-8')
    version_match = re.search(r'release\s+([\d\.]+)', version_str)
    if not version_match:
        logging.warning('Failed to extract CUDA version from nvcc output')
        return None

    cuda_ver = version_match.group(1)
    major, minor = map(int, cuda_ver.split('.'))
    cuda_ver_int = 10 * major + minor

    # Set CUDA environment variables
    cuda_home = os.environ.get('CUDA_HOME', '')
    if not cuda_home:
        cuda_home = os.environ['CONDA_PREFIX']
        if not os.path.isdir(cuda_home):
            cuda_home = ''
    if not cuda_home:
        logging.warning('CUDA_HOME is not set. Please set it to enable CUDA extensions.')
        return None

    cudaconfig['home'] = cuda_home
    cudaconfig['nvcc'] = os.path.join(cuda_home, 'bin', 'nvcc')
    cudaconfig['include'] = os.path.join(cuda_home, 'include')
    cudaconfig['lib64'] = os.path.join(cuda_home, 'lib64')

    # Set CUDA version-specific post-args
    specs = get_cuda_device_specs()
    arch = str(specs["compute_capability"][0]) + str(specs["compute_capability"][1])
    post_args = [f"-arch=sm_{arch}", "--ptxas-options=-v", "-O2"]

    if sys.platform == "win32":
        cudaconfig['lib64'] = os.path.join(cuda_home, 'lib', 'x64')
        post_args += ['-Xcompiler', '/MD', '-std=c++14',  "-Xcompiler", "/openmp"]
        if HALF_PRECISION:
            post_args += ["-Xcompiler", "/D HALF_PRECISION"]
    else:
        post_args += ['-c', '--compiler-options', "'-fPIC'",
            "--compiler-options", "'-std=c++14'"]
        if HALF_PRECISION:
            post_args += ["--compiler-options", "'-D HALF_PRECISION'"]

    cudaconfig['post_args'] = post_args

    return cudaconfig


# This code to build .cu extensions with nvcc is taken from cupy:
# https://github.com/cupy/cupy/blob/master/cupy_setup_build.py
class _UnixCCompiler(unixccompiler.UnixCCompiler):
  src_extensions = list(unixccompiler.UnixCCompiler.src_extensions)
  src_extensions.append('.cu')

  def _compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
    # For sources other than CUDA C ones, just call the super class method.
    if os.path.splitext(src)[1] != '.cu':
      return unixccompiler.UnixCCompiler._compile(
        self, obj, src, ext, cc_args, extra_postargs, pp_opts)

    # For CUDA C source files, compile them with NVCC.
    _compiler_so = self.compiler_so
    try:
      nvcc_path = CUDA['nvcc']
      post_args = CUDA['post_args']
      # TODO? base_opts = build.get_compiler_base_options()
      self.set_executable('compiler_so', nvcc_path)

      return unixccompiler.UnixCCompiler._compile(
        self, obj, src, ext, cc_args, post_args, pp_opts)
    finally:
      self.compiler_so = _compiler_so


class _MSVCCompiler(msvccompiler.MSVCCompiler):
  _cu_extensions = ['.cu']

  src_extensions = list(unixccompiler.UnixCCompiler.src_extensions)
  src_extensions.extend(_cu_extensions)

  def _compile_cu(self, sources, output_dir=None, macros=None,
          include_dirs=None, debug=0, extra_preargs=None,
          extra_postargs=None, depends=None):
    # Compile CUDA C files, mainly derived from UnixCCompiler._compile().
    macros, objects, extra_postargs, pp_opts, _build = \
      self._setup_compile(output_dir, macros, include_dirs, sources,
                depends, extra_postargs)

    compiler_so = CUDA['nvcc']
    cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)
    post_args = CUDA['post_args']

    for obj in objects:
      try:
        src, _ = _build[obj]
      except KeyError:
        continue
      try:
        self.spawn([compiler_so] + cc_args + [src, '-o', obj] + post_args)
      except errors.DistutilsExecError as e:
        raise errors.CompileError(str(e))

    return objects

  def compile(self, sources, **kwargs):
    # Split CUDA C sources and others.
    cu_sources = []
    other_sources = []
    for source in sources:
      if os.path.splitext(source)[1] == '.cu':
        cu_sources.append(source)
      else:
        other_sources.append(source)

    # Compile source files other than CUDA C ones.
    other_objects = msvccompiler.MSVCCompiler.compile(
      self, other_sources, **kwargs)

    # Compile CUDA C sources.
    cu_objects = self._compile_cu(cu_sources, **kwargs)

    # Return compiled object filenames.
    return other_objects + cu_objects


class CudaBuildExt(setuptools_build_ext):
  """Custom `build_ext` command to include CUDA C source files."""

  def run(self):
    if CUDA is not None:
      def wrap_new_compiler(func):
        def _wrap_new_compiler(*args, **kwargs):
          try:
            return func(*args, **kwargs)
          except errors.DistutilsPlatformError:
            if sys.platform != 'win32':
              CCompiler = _UnixCCompiler
            else:
              CCompiler = _MSVCCompiler
            return CCompiler(
              None, kwargs['dry_run'], kwargs['force'])
        return _wrap_new_compiler
      ccompiler.new_compiler = wrap_new_compiler(ccompiler.new_compiler)
      # Intentionally causes DistutilsPlatformError in
      # ccompiler.new_compiler() function to hook.
      self.compiler = 'nvidia'

    setuptools_build_ext.run(self)


CUDA = prepare_cuda()
assert CUDA is not None
BUILDEXT = CudaBuildExt if CUDA else setuptools_build_ext
