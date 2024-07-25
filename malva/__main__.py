import logging
from rich.logging import RichHandler
import os

from malva.cli import cmdline_main

def run_malva():
    KATOSTE_DEBUG = os.environ.get('KATOSTE_DEBUG', '0')
    _level = logging.DEBUG if KATOSTE_DEBUG == '1' else logging.INFO
    logging.basicConfig(format="%(message)s", datefmt="[%X]", level=_level, handlers=[RichHandler(enable_link_path=False, show_path=False)])
    cmdline_main()


if __name__ == "__main__":
    run_malva()
