import numpy
from distutils.command.build_ext import build_ext
from setuptools import Extension, setup

unordered_dense_include = "./include"

class BuildExt(build_ext):
    def build_extensions(self):
        try:
            super().build_extensions()
        except Exception:
            pass

extensions = [
    Extension(
        "katoste.kmer_processing",
        ["katoste/kmer_processing.pyx"],
        include_dirs=[unordered_dense_include, numpy.get_include()],
        language="c++",
        extra_compile_args=["-std=c++17"],
    ),
    Extension(
        "katoste.fastq_processing",
        ["katoste/fastq_processing.pyx"],
        include_dirs=[unordered_dense_include, numpy.get_include()],
        language="c++",
        extra_compile_args=["-std=c++17"],
    ),
    Extension(
        "katoste.reader",
        ["katoste/reader.pyx"],
        include_dirs=[unordered_dense_include, numpy.get_include()],
        language="c++",
        extra_compile_args=["-std=c++17"],
    ),
    Extension(
        "katoste.faster_classes",
        ["katoste/faster_classes.pyx"],
        include_dirs=[unordered_dense_include, numpy.get_include()],
        language="c++",
        extra_compile_args=["-std=c++17"],
    ),
    Extension(
        "katoste.fast_map",
        ["katoste/fast_map.pyx"],
        include_dirs=[unordered_dense_include, numpy.get_include()],
        language="c++",
        extra_compile_args=["-std=c++17"],
    ),
]

def build(setup_kwargs):
    from Cython.Build import cythonize
    setup_kwargs.update(
        dict(
            cmdclass=dict(build_ext=BuildExt),
            ext_modules=cythonize(extensions),
        )
    )

if __name__ == "__main__":
    setup(
        name="katoste",
        # Add other necessary setup parameters
        ext_modules=extensions,
        cmdclass=dict(build_ext=BuildExt),
    )