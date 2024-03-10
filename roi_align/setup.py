from __future__ import print_function
import os
import torch
from pkg_resources import parse_version

min_version = parse_version('1.0.0')
current_version = parse_version(torch.__version__)


if current_version < min_version: #PyTorch before 1.0
    from torch.utils.ffi import create_extension

    sources = ['src/roi_align.c']
    headers = ['src/roi_align.h']
    extra_objects = []
    #sources = []
    #headers = []
    defines = []
    with_cuda = False

    this_file = os.path.dirname(os.path.realpath(__file__))
    print(this_file)

    if torch.cuda.is_available():
        print('Including CUDA code.')
        sources += ['src/roi_align_cuda.c']
        headers += ['src/roi_align_cuda.h']
        defines += [('WITH_CUDA', None)]
        with_cuda = True
    
        extra_objects = ['src/roi_align_kernel.cu.o']
        extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

    ffi = create_extension(
        '_ext.roi_align',
        headers=headers,
        sources=sources,
        define_macros=defines,
        relative_to=__file__,
        with_cuda=with_cuda,
        extra_objects=extra_objects
    )

    if __name__ == '__main__':
        ffi.build()
else: # PyTorch 1.0 or later
    from setuptools import setup
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    print('Including CUDA code.')
    current_dir = os.path.dirname(os.path.realpath(__file__))
    #cuda_include = '/usr/local/cuda-10.0/include'

    #GPU version
    setup(
        name='roi_align_api',
        ext_modules=[
            CUDAExtension(
                    name='roi_align_api',
                    sources=['src/roi_align_cuda.cpp', 'src/roi_align_kernel.cu'],
                    include_dirs=[current_dir]+torch.utils.cpp_extension.include_paths(cuda=True)
                    )
        ],
        cmdclass={
            'build_ext': BuildExtension
        })
