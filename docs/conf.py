# -- Project information -----------------------------------------------------
project = 'PyBic'
copyright = '2023-2026, G. Riggs and T. Matheny'
author = 'Greg Riggs'

# The full version, including alpha/beta/rc tags
release = '0.1.0'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. 
# Common extensions include:
# 'sphinx.ext.autodoc' - to generate docs from docstrings
# 'sphinx.ext.napoleon' - for Google/NumPy style docstrings
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages. 
# 'sphinx_rtd_theme' is the standard theme for Read the Docs.
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. 
html_static_path = ['_static']
