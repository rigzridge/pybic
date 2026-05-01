# -- Project information -----------------------------------------------------
project = 'PyBic'
copyright = '2022-2026, G. Riggs and T. Matheny'
author = 'Greg Riggs'

# Version #
release = '2.1.0'

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
import matplotlib.pyplot as plt
import inspect
import pybic

# -- General configuration ---------------------------------------------------

# Sphinx extension modules
# 'sphinx.ext.autodoc' - to generate docs from docstrings
# 'sphinx.ext.napoleon' - for Google/NumPy style docstrings
extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'nbsphinx',
    'sphinx.ext.linkcode',
    'sphinx_copybutton',
    # 'matplotlib.sphinxext.plot_directive',
]
source_suffix = ['.rst', '.md']

def linkcode_resolve(domain, info):
    if domain != 'py' or not info['module']:
        return None
    
    module = info['module']
    fxn = info['fullname']
    github_repo = "https://github.com/rigzridge/pybic/blob/main"

    try:
        # Try to get source code of function or method
        source, lineno = inspect.getsourcelines(eval('%s.%s' % (module, fxn)))
    except:
        # Attributes throw errors so first link to BicAn
        source, lineno = inspect.getsourcelines(pybic.BicAn)
        # Try to get attributes
        # for i, line in enumerate(source):
        #     if f'{fxn}' in line:
                
    lineno = '#L%d' % lineno

    return '%s/%s.py%s' % (github_repo, module, lineno)

intersphinx_mapping = {
    'rtd': ('https://docs.readthedocs.io/en/stable/', None),
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    
}
intersphinx_disabled_domains = ['std']

autodoc_mock_imports = ['os','numpy','matplotlib','scipy','tkinter','datetime','mpl_toolkits']

# Should generate autosummary?
autosummary_generate = True

# Include TODO sections
todo_include_todos = True

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
