from distutils.command.build_ext import build_ext

class BuildExt(build_ext):
    def build_extensions(self):
        try:
            super().build_extensions()
        except Exception:
            pass


def build(setup_kwargs):
    from Cython.Build import cythonize
    setup_kwargs.update(
        dict(
            cmdclass=dict(build_ext=BuildExt),
            ext_modules=cythonize(["katoste/kmer_processing.pyx", "katoste/fastq_processing.pyx", "katoste/reader.pyx"]),
        )
    )