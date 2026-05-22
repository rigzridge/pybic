.. _home:

====================
PyBic documentation!
====================

.. include:: ../README.rst

.. hint::
   :collapsible:

   This hint is collapsible, but initially open.
   :ref:`This links back to the top! <home>`

Getting started
----------------

.. tabs::

   .. code-tab:: bash

         python3 pybic.py

   .. code-tab:: py

         import pybic as bic
         b = bic.RunDemo()

.. note::

   While the :mod:`pybic` module is relatively complete,
   this documentation is under active development!
   Updates should be coming throughout 2026...

.. todo:: 

   Include base units for time-series


.. toctree::
   :caption: Sections
   :hidden:

   Home <self>
   theory
   plotting
   api


.. autosummary::
   :toctree: generated
   :caption: Module

   pybic
