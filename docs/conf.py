# -- Project information -----------------------------------------------------
project = 'PyBic'
copyright = '2022-2026, G. Riggs and T. Matheny'
author = 'Greg Riggs'

# Version #
release = '2.1.0'

# -- General configuration ---------------------------------------------------

# Sphinx extension modules
# 'sphinx.ext.autodoc' - to generate docs from docstrings
# 'sphinx.ext.napoleon' - for Google/NumPy style docstrings
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
]

# Add any paths that contain templates here, relative to this directory.
# Not really using this yet...
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# ... or this lol
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages. 
# 'sphinx_rtd_theme' is the standard theme for Read the Docs.
# Will not work w/o this!
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. 
# Not used yet!
html_static_path = ['_static']
