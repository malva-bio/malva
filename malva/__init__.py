try:
    from importlib.metadata import version, PackageNotFoundError
    __version__ = version("malva")
except PackageNotFoundError:
    # Package not yet installed (e.g., running from source tree)
    __version__ = "unknown"

from malva.malva_index import MalvaIndex