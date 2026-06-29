API
====

The central class
-----------------

.. autosummary::
   :toctree: generated

   pybic.BicAn

Demonstration
-----------------

.. autosummary::
   :toctree: generated

   pybic.RunDemo

Signal generation
-----------------

.. autosummary::
   :toctree: generated

   pybic.SignalGen
   pybic.TestSignal

Instantaneous frequency analysis
--------------------------------

.. autosummary::
   :toctree: generated

   pybic.WhittakerShannon
   pybic.InstFreqZeroCross

Pre-processing
--------------

.. autosummary::
   :toctree: generated

   pybic.ApplyDetrend
   pybic.ApplyBandpass
   pybic.ApplyRealBandpass
   pybic.ApplySimpleFilter

Time-frequency representations
------------------------------

.. autosummary::
   :toctree: generated

   pybic.HannWindow
   pybic.FlatTopWindow
   pybic.ApplyCWT
   pybic.ApplySTFT
   pybic.CalcHistVsT

Full polyspectra
----------------

.. autosummary::
   :toctree: generated

   pybic.SpecToCoherence
   pybic.SpecToBispec
   pybic.SpecToCrossBispec
   pybic.SpecToTrispec

Local polyspectra
-----------------

.. autosummary::
   :toctree: generated

   pybic.GetBispec
   pybic.GetBispecBootstrap
   pybic.GetPolySpec

Plot aids
----------

.. autosummary::
   :toctree: generated

   pybic.Plot
   pybic.PlotLabels
   pybic.PlotRHS
   pybic.PlotTop
   pybic.PlotTimeline
   pybic.DrawSimplex
   pybic.ScaleToString

Helpers
-------

.. autosummary::
   :toctree: generated

   pybic.LoadBar
   pybic.FileDialog
   pybic.nRandSumLessThanUnity


Extras
------

.. autosummary::
   :toctree: generated

   pybic.arrmin
   pybic.bin_mat
   pybic.boxcar_ave
   pybic.diff_to_sum_vec
   pybic.dphase_dt