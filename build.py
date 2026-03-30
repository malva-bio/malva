# Copyright (c) 2025 Daniel León-Periñán and Nikolaos Karaiskos
#                    Rajewsky Lab, Max Delbrück Center for Molecular Medicine (MDC), Berlin
#
# Non-commercial and academic use only. See LICENSE for full terms.

import os
import sys
import numpy
from distutils.command.build_ext import build_ext
from setuptools import Extension
from Cython.Build import cythonize


def _has_liburing():
    """Return True if liburing is available on this Linux system."""
    if sys.platform != "linux":
        return False
    try:
        import pkgconfig
        return pkgconfig.exists("liburing")
    except Exception:
        pass
    # fallback: check header directly
    import subprocess
    try:
        r = subprocess.run(
            ["gcc", "-x", "c", "-", "-o", "/dev/null", "-luring"],
            input=b"#include <liburing.h>\nint main(){return 0;}\n",
            capture_output=True,
        )
        return r.returncode == 0
    except Exception:
        return False

# Set MALVA_DEBUG_BUILD=1 to build with profiling hooks and debug symbols.
# Default (0) builds with full production optimisations for distribution.
DEBUG_BUILD = os.environ.get('MALVA_DEBUG_BUILD', '0') == '1'

unordered_dense_include = "./include"


class BuildExt(build_ext):
    def build_extensions(self):
        try:
            super().build_extensions()
        except Exception:
            pass


def get_extensions():
    if DEBUG_BUILD:
        compiler_directives = {
            'language_level': 3,
            'boundscheck': False,
            'wraparound': False,
            'initializedcheck': False,
            'cdivision': True,
            'linetrace': True,
            'profile': True,
        }
        compile_args = ["-std=c++17"]
        link_args = []
        macros = [('CYTHON_TRACE', '1'), ('CYTHON_TRACE_NOGIL', '1')]
    else:
        compiler_directives = {
            'language_level': 3,
            'boundscheck': False,
            'wraparound': False,
            'initializedcheck': False,
            'cdivision': True,
            'linetrace': False,
            'profile': False,
            'embedsignature': False,
            'emit_code_comments': False,
        }
        compile_args = [
            "-std=c++17",
            "-O3",
            "-ffast-math",
            "-DNDEBUG",
            "-fvisibility=hidden",
            "-ffunction-sections",
            "-fdata-sections",
        ]
        macros = [('NDEBUG', '1')]

        if sys.platform == 'linux':
            # GCC-specific: avoid PLT indirection for intra-DSO calls
            compile_args.append("-fno-semantic-interposition")
            link_args = [
                "-Wl,--strip-all",        # strip debug symbols
                "-Wl,--gc-sections",      # drop unused code/data sections
                "-Wl,--exclude-libs,ALL", # hide symbols from static libs
                "-Wl,-z,relro",           # read-only relocations (security)
                "-Wl,-z,now",             # full RELRO
            ]
        else:
            # macOS (Apple ld)
            link_args = ["-Wl,-dead_strip"]

    common_include_dirs = [unordered_dense_include, numpy.get_include()]

    extensions = [
        Extension(
            "malva.indexes",
            ["malva/indexes.pyx"],
            include_dirs=common_include_dirs,
            language="c++",
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            define_macros=macros,
            libraries=["uring"] if _has_liburing() else [],
        ),
        Extension(
            "malva.barcodes",
            ["malva/barcodes.pyx"],
            include_dirs=common_include_dirs,
            language="c++",
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            define_macros=macros,
        ),
        Extension(
            "malva.kmer_processing",
            ["malva/kmer_processing.pyx"],
            include_dirs=common_include_dirs,
            language="c++",
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            define_macros=macros,
        ),
        Extension(
            "malva.fastq_processing",
            ["malva/fastq_processing.pyx"],
            include_dirs=common_include_dirs,
            language="c++",
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            define_macros=macros,
        ),
        Extension(
            "malva.reader",
            ["malva/reader.pyx"],
            include_dirs=common_include_dirs,
            language="c++",
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            define_macros=macros,
        ),
        Extension(
            "malva.fast_map",
            ["malva/fast_map.pyx"],
            include_dirs=common_include_dirs,
            language="c++",
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            define_macros=macros,
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
