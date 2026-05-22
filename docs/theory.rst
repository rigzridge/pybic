Theory
======

The bispectrum
--------------

.. math::

	\mathcal{B}_{xyz}(f_1,f_2) = \langle X(f_1)Y(f_2)Z^*(f_1+f_2) \rangle, 

where :math:`x, y, z` are time series with 
corresponding Fourier transforms :math:`X, Y, Z`,
and :math:`\langle ... \rangle` denotes time averaging.

The (squared) bicoherence spectrum
----------------------------------

.. math::
	
	b^2_{xyz}(f_1,f_2) = \frac{|B_{xyz}(f_1,f_2)|^2 }{ \left\langle|X(f_1)Y(f_2)|^2\right\rangle 
	\left\langle|Z(f_1+f_2)|^2\right\rangle + \varepsilon },

where :math:`\varepsilon` is a small number meant to prevent 0/0 = ``NaN`` catastrophe.