"""
Adapted from xopen: Open compressed files transparently.

This modifies slightly so we can also use "bgzip"
for faster reading of fastq.gz files
"""

import io
from typing import (
    Optional,
    IO,
    overload,
    BinaryIO,
    Literal,
)

from xopen import (
    _PROGRAM_SETTINGS, 
    XOPEN_DEFAULT_GZIP_COMPRESSION, 
    _PipedCompressionProgram, 
    _ProgramSettings, 
    _filepath_from_path_or_filelike,
    _file_or_path_to_binary_stream,
    _open_stdin_or_out,
    _file_is_a_socket_or_pipe,
    _detect_format_from_content,
    _detect_format_from_extension,
    _open_bz2,
    _open_reproducible_gzip,
    _open_xz,
    _open_zst,
    _available_cpu_count,
    FileOrPath,
    igzip_threaded,
    gzip_ng_threaded,
    zlib_ng,
)

_PROGRAM_SETTINGS["bgzip"] = _ProgramSettings(("bgzip",), tuple(range(0, 10)), "-@")

@overload
def xopen(
    filename: FileOrPath,
    mode: Literal["r", "w", "a", "rt", "wt", "at"] = ...,
    compresslevel: Optional[int] = ...,
    threads: Optional[int] = ...,
    *,
    encoding: str = ...,
    errors: Optional[str] = ...,
    newline: Optional[str] = ...,
    format: Optional[str] = ...,
) -> io.TextIOWrapper:
    ...


@overload
def xopen(
    filename: FileOrPath,
    mode: Literal["rb", "wb", "ab"],
    compresslevel: Optional[int] = ...,
    threads: Optional[int] = ...,
    *,
    encoding: str = ...,
    errors: None = ...,
    newline: None = ...,
    format: Optional[str] = ...,
) -> BinaryIO:
    ...


def xopen(  # noqa: C901
    filename: FileOrPath,
    mode: Literal["r", "w", "a", "rt", "rb", "wt", "wb", "at", "ab"] = "r",
    compresslevel: Optional[int] = None,
    threads: Optional[int] = None,
    *,
    encoding: str = "utf-8",
    errors: Optional[str] = None,
    newline: Optional[str] = None,
    format: Optional[str] = None,
) -> IO:
    """
    A replacement for the "open" function that can also read and write
    compressed files transparently. The supported compression formats are gzip,
    bzip2, xz and zstandard. If the filename is '-', standard output (mode 'w') or
    standard input (mode 'r') is returned. Filename can be a string or a
    file object. (See https://docs.python.org/3/glossary.html#term-file-object.)

    When writing, the file format is chosen based on the file name extension:
    - .gz uses gzip compression
    - .bz2 uses bzip2 compression
    - .xz uses xz/lzma compression
    - .zst uses zstandard compression
    - otherwise, no compression is used

    When reading, if a file name extension is available, the format is detected
    using it, but if not, the format is detected from the contents.

    mode can be: 'rt', 'rb', 'at', 'ab', 'wt', or 'wb'. Also, the 't' can be omitted,
    so instead of 'rt', 'wt' and 'at', the abbreviations 'r', 'w' and 'a' can be used.

    compresslevel is the compression level for writing to gzip, xz and zst files.
    This parameter is ignored for the other compression formats.
    If set to None, a default depending on the format is used:
    gzip: 6, xz: 6, zstd: 3.

    When threads is None (the default), compressed file formats are read or written
    using a pipe to a subprocess running an external tool such as,
    ``pbzip2``, ``gzip`` etc., see PipedGzipWriter, PipedGzipReader etc.
    If the external tool supports multiple threads, *threads* can be set to an int
    specifying the number of threads to use.
    If no external tool supporting the compression format is available, the file is
    opened calling the appropriate Python function
    (that is, no subprocess is spawned).

    Set threads to 0 to force opening the file without using a subprocess.

    encoding, errors and newline are used when opening a file in text mode.
    The parameters have the same meaning as in the built-in open function,
    except that the default encoding is always UTF-8 instead of the
    preferred locale encoding.

    format overrides the autodetection of input and output formats. This can be
    useful when compressed output needs to be written to a file without an
    extension. Possible values are "gz", "xz", "bz2", "zst".
    """
    if mode in ("r", "w", "a"):
        mode += "t"  # type: ignore
    if mode not in ("rt", "rb", "wt", "wb", "at", "ab"):
        raise ValueError("Mode '{}' not supported".format(mode))
    binary_mode = mode[0] + "b"
    filepath = _filepath_from_path_or_filelike(filename)

    # Open non-regular files such as pipes and sockets here to force opening
    # them once.
    if filename == "-":
        filename = _open_stdin_or_out(binary_mode)
    elif _file_is_a_socket_or_pipe(filename):
        filename = open(filename, binary_mode)  # type: ignore

    if format not in (None, "gz", "xz", "bz2", "zst"):
        raise ValueError(
            f"Format not supported: {format}. "
            f"Choose one of: 'gz', 'xz', 'bz2', 'zst'"
        )
    detected_format = format or _detect_format_from_extension(filepath)
    if detected_format is None and "r" in mode:
        detected_format = _detect_format_from_content(filename)

    if detected_format == "gz":
        opened_file = _open_gz(filename, binary_mode, compresslevel, threads)
    elif detected_format == "xz":
        opened_file = _open_xz(filename, binary_mode, compresslevel, threads)
    elif detected_format == "bz2":
        opened_file = _open_bz2(filename, binary_mode, compresslevel, threads)
    elif detected_format == "zst":
        opened_file = _open_zst(filename, binary_mode, compresslevel, threads)
    else:
        opened_file, _ = _file_or_path_to_binary_stream(filename, binary_mode)

    if "t" in mode:
        return io.TextIOWrapper(opened_file, encoding, errors, newline)
    return opened_file


def _open_gz(
    filename: FileOrPath,
    mode: str,
    compresslevel: Optional[int],
    threads: Optional[int],
):
    """
    Open a gzip file. The ISA-L library is preferred when applicable because
    it is the fastest. Then zlib-ng which is not as fast, but supports all
    compression levels. After that comes pigz, which can utilize multiple
    threads and is more efficient than gzip, even on one core. gzip is chosen
    when none of the alternatives are available. Despite it being able to use
    only one core, it still finishes faster than using the builtin gzip library
    as the (de)compression is moved to another thread.
    """
    assert mode in ("rb", "ab", "wb")
    if compresslevel is None:
        # Force the same compression level on every tool regardless of
        # library defaults
        compresslevel = XOPEN_DEFAULT_GZIP_COMPRESSION
    if compresslevel not in range(10):
        # Level 0-9 are supported regardless of backend support
        # (zlib_ng supports -1, pigz supports 11 etc.)
        raise ValueError(
            f"gzip compresslevel must be in range 0-9, got {compresslevel}."
        )

    if threads != 0:
        # First we try with bgzip, which can be the fastest for fastq.gz files
        for program in ("bgzip", ):
            try:
                return _PipedCompressionProgram(
                    filename,
                    mode,
                    compresslevel,
                    threads,
                    _PROGRAM_SETTINGS[program],
                )
            # ValueError when compresslevel is not supported. i.e. gzip and level 0
            except (OSError, ValueError, KeyError):
                pass  # We try without threads.

        # Igzip level 0 does not output uncompressed deflate blocks as zlib does
        # and level 3 is slower but does not compress better than level 1 and 2.
        if igzip_threaded and (compresslevel in (1, 2) or "r" in mode):
            return igzip_threaded.open(  # type: ignore
                filename,
                mode,
                compresslevel,
                threads=1,
            )
        if gzip_ng_threaded and zlib_ng:
            return gzip_ng_threaded.open(
                filename,
                mode,
                # zlib-ng level 1 is 50% bigger than zlib level 1. Level
                # 2 gives a size close to expectations.
                compresslevel=2 if compresslevel == 1 else compresslevel,
                threads=threads or max(_available_cpu_count(), 4),
            )
        
        # Lastly the other two last options if none of the above is available
        for program in ("pigz", "gzip"):
            try:
                return _PipedCompressionProgram(
                    filename,
                    mode,
                    compresslevel,
                    threads,
                    _PROGRAM_SETTINGS[program],
                )
            # ValueError when compresslevel is not supported. i.e. gzip and level 0
            except (OSError, ValueError):
                pass  # We try without threads.

    return _open_reproducible_gzip(filename, mode=mode, compresslevel=compresslevel)
