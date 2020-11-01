from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# Python interface
setup(
    name='PyTorchStructuralLosses',
    ext_modules=[
        CUDAExtension(
            name='StructuralLossesBackend',
            sources=[
                'structural_loss.cpp',
                'approxmatch.cu',
                'nndistance.cu'
            ],
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
