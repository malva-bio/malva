import numpy
from distutils.command.build_ext import build_ext
from setuptools import Extension
from Cython.Build import cythonize

unordered_dense_include = "./include"

class BuildExt(build_ext):
    def build_extensions(self):
        try:
            super().build_extensions()
        except Exception:
            pass

def get_extensions():
    compiler_directives = {
        'language_level': 3,
        'boundscheck': False,
        'wraparound': False,
        'initializedcheck': False,
        'cdivision': True,
        'linetrace': True,
        'profile': True,
    }

    extensions = [
        Extension(
            "malva.kmer_processing",
            ["malva/kmer_processing.pyx"],
            include_dirs=[unordered_dense_include, numpy.get_include()],
            language="c++",
            extra_compile_args=["-std=c++17"],
            define_macros=[('CYTHON_TRACE', '1'),
                         ('CYTHON_TRACE_NOGIL', '1')],
        ),
        Extension(
            "malva.fastq_processing",
            ["malva/fastq_processing.pyx"],
            include_dirs=[unordered_dense_include, numpy.get_include()],
            language="c++",
            extra_compile_args=["-std=c++17"],
            define_macros=[('CYTHON_TRACE', '1'),
                         ('CYTHON_TRACE_NOGIL', '1')],
        ),
        Extension(
            "malva.reader",
            ["malva/reader.pyx"],
            include_dirs=[unordered_dense_include, numpy.get_include()],
            language="c++",
            extra_compile_args=["-std=c++17"],
            define_macros=[('CYTHON_TRACE', '1'),
                         ('CYTHON_TRACE_NOGIL', '1')],
        ),
        Extension(
            "malva.indexes",
            ["malva/indexes.pyx"],
            include_dirs=[unordered_dense_include, numpy.get_include()],
            language="c++",
            extra_compile_args=["-std=c++17"],
            define_macros=[('CYTHON_TRACE', '1'),
                         ('CYTHON_TRACE_NOGIL', '1')],
        ),
        Extension(
            "malva.fast_map",
            ["malva/fast_map.pyx"],
            include_dirs=[unordered_dense_include, numpy.get_include()],
            language="c++",
            extra_compile_args=["-std=c++17"],
            define_macros=[('CYTHON_TRACE', '1'),
                         ('CYTHON_TRACE_NOGIL', '1')],
        ),
    ]
    
    return cythonize(extensions, 
                    compiler_directives=compiler_directives,
                    force=True)

def build(setup_kwargs):
    setup_kwargs.update(
        dict(
            cmdclass=dict(build_ext=BuildExt),
            ext_modules=get_extensions(),
        )
    )

if __name__ == "__main__":
    from setuptools import setup
    setup(
        name="malva",
        ext_modules=get_extensions(),
        cmdclass=dict(build_ext=BuildExt),
    )