import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# Set up pandoc from pypandoc_binary if available
try:
    import pypandoc
    # This ensures pypandoc uses its bundled pandoc
    pypandoc.get_pandoc_path()
except (ImportError, OSError):
    print("Warning: pypandoc/pandoc not found. Install with: conda install -c conda-forge pandoc")

project = 'Malva'
copyright = '2025, Malva Team'
author = 'Malva Team'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinxarg.ext',
    'sphinx_autodoc_typehints',
    'myst_parser',
    'sphinx_click',
    'sphinx_copybutton',
    'nbsphinx',
]

# nbsphinx settings
nbsphinx_execute = 'never'  # Don't re-execute notebooks, use stored outputs

# FURO THEME
html_theme = 'furo'
html_title = "Malva"
html_logo = "_static/malva_logo.svg"

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "source_repository": "https://github.com/malva_bio/malva/",
    "source_branch": "main",
    "source_directory": "docs/",
}

# Static files
html_static_path = ['_static']

# Settings
suppress_warnings = ['autodoc.duplicate_object']
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True
source_suffix = {'.rst': None, '.md': 'myst_parser'}
master_doc = 'index'