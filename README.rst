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

This module implements *Bicoherence Analyzer* in Python!

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

Documentation
-------------

Check out our (in-development) `Read the Docs page`_ for detailed information about the :mod:`pybic` module and the associated :obj:`pybic.BicAn` class.

Tutorials
---------

For convenience, the following Jupyter notebooks demonstrate many of the features of PyBic:

* `Guided tour`_
* |bic.Plot|_

Citing our work
---------------

To reference PyBic/BicAn, please cite our publication in *Computer Physics Communications*, `BicAn: An integrated, open-source framework for polyspectral analysis`_.

*Note that the preprint is available above!*

Theory
------

The bispectrum
^^^^^^^^^^^^^^

.. math::

	\mathcal{B}_{xyz}(f_1,f_2) = \langle X(f_1)Y(f_2)Z^*(f_1+f_2) \rangle, 

where :math:`x, y, z` are time series with 
corresponding Fourier transforms :math:`X, Y, Z`,
and :math:`\langle ... \rangle` denotes time averaging.

The (squared) bicoherence spectrum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::
	
	b^2_{xyz}(f_1,f_2) = \frac{|B_{xyz}(f_1,f_2)|^2 }{ \left\langle|X(f_1)Y(f_2)|^2\right\rangle 
	\left\langle|Z(f_1+f_2)|^2\right\rangle + \varepsilon },

where :math:`\varepsilon` is a small number meant to prevent 0/0 = ``NaN`` catastrophe.

Contact
--------

Please reach out with any questions to rigzridge@gmail.com!

.. _Guided Tour:
	https://colab.research.google.com/drive/1GnJddGDVVIWK44B-_0Mfoe-tLKWoXFrb?usp=sharing

.. |bic.Plot| replace:: ``bic.Plot()`` documentation
.. _bic.Plot:
	https://colab.research.google.com/drive/1NJmjnkhD9wWd_uYRYDWSOEatzS_5Nzm3?usp=sharing

.. _BicAn\: An integrated, open-source framework for polyspectral analysis:
    https://doi.org/10.1016/j.cpc.2026.110097

.. _Read the Docs page:
	https://pybic.readthedocs.io/en/latest/
