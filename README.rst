.. image:: https://raw.githubusercontent.com/rigzridge/pybic/refs/heads/main/PyBic.png
   :alt: PyBic logo

.. image:: https://img.shields.io/github/commit-activity/m/rigzridge/pybic
    :alt: GitHub commit activity

.. image:: https://readthedocs.org/projects/pybic/badge/?version=latest
    :target: https://pybic.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/github/license/rigzridge/pybic
    :target: https://www.gnu.org/licenses/gpl-3.0
    :alt: License: GPL v3

.. image:: https://img.shields.io/github/languages/code-size/rigzridge/pybic?color=purple
    :alt: Code size

**PyBic** is an open-source module specializing in signal processing, 
with particular emphasis on polyspectral analysis. 

PyBic's *Bicoherence Analyzer* class (|pybic.BicAn|_) simplifies the
estimation, visualization, and interpretation of low order polyspectra, 
*e.g.*, the cross-spectrum, bispectrum, and trispectrum.

Check out the `Theory`_ docs for a brief intro to polyspectra!

Quick start
-----------

To get rolling, place the ``pybic.py`` file in your desired directory, and try

.. code-block:: python

	import pybic as bic
	b = bic.BicAn('demo')

Alternatively, to analyze a time-series ``x`` sampled at ``fS``, use

.. code-block:: python

	import pybic as bic
	b = bic.BicAn(x,samprate=fS)

See our `Getting started`_ section for more info!

Tutorials
---------

Our (in-development) `Read the Docs page`_ provides detailed information about the |pybic|_ module and the associated |pybic.BicAn|_ class.

For convenience, we've developed a number of `Examples`_ to get a feel for the module, and the following Jupyter notebooks demonstrate many of PyBic's features:

* `Guided tour`_
* |pybic.Plot documentation|_

Citing our work
---------------

To reference PyBic/BicAn, please cite our publication in *Computer Physics Communications*, `BicAn: An integrated, open-source framework for polyspectral analysis`_.

Note that the `preprint`_ is freely available!

Contact
--------

Have a question, comment, or critique? Feel free to reach out to rigzridge@gmail.com!

.. _Guided Tour:
	https://colab.research.google.com/drive/1GnJddGDVVIWK44B-_0Mfoe-tLKWoXFrb?usp=sharing

.. |pybic.Plot documentation| replace:: ``pybic.Plot()`` documentation
.. _pybic.Plot documentation:
	https://colab.research.google.com/drive/1NJmjnkhD9wWd_uYRYDWSOEatzS_5Nzm3?usp=sharing

.. |pybic| replace:: ``pybic``
.. _pybic:
	https://pybic.readthedocs.io/en/latest/generated/pybic.html

.. |pybic.BicAn| replace:: ``pybic.BicAn``
.. _pybic.BicAn:
	https://pybic.readthedocs.io/en/latest/generated/pybic.BicAn.html

.. _BicAn\: An integrated, open-source framework for polyspectral analysis:
    https://doi.org/10.1016/j.cpc.2026.110097

.. _Read the Docs page:
	https://pybic.readthedocs.io/en/latest/

.. _Getting started:
	https://pybic.readthedocs.io/en/latest/start.html

.. _Examples:
	https://pybic.readthedocs.io/en/latest/examples.html

.. _Theory:
	https://pybic.readthedocs.io/en/latest/theory.html

.. _preprint:
	https://github.com/rigzridge/pybic/blob/main/BicAn_An%20integrated%2C%20open-source%20framework%20for%20polyspectral%20analysis_PREPRINT.pdf
