Theory
======

The bispectrum
--------------

.. math::
	:label: test_B

	\mathcal{B}_{xyz}(f_1,f_2) = \langle X(f_1)Y(f_2)Z^*(f_1+f_2) \rangle, 

where :math:`x, y, z` are time series with 
corresponding Fourier transforms :math:`X, Y, Z`,
and :math:`\langle ... \rangle` denotes time averaging.

The (squared) bicoherence spectrum
----------------------------------

.. math::
	:label: b2_def
	
	b^2_{xyz}(f_1,f_2) = \frac{|\mathcal{B}_{xyz}(f_1,f_2)|^2 }{ \left\langle|X(f_1)Y(f_2)|^2\right\rangle 
	\left\langle|Z(f_1+f_2)|^2\right\rangle + \varepsilon },

where :math:`\varepsilon` is a small number meant to prevent 0/0 = ``NaN`` catastrophe,
and the bispectrum is given by :eq:`test_B`.

Instantaneous difference frequency
----------------------------------

Here we define 

.. math::

    \Delta f_{\rm inst}(t) \equiv \frac{1}{2\pi}\frac{d\varsigma(t)}{dt},

where the n-phase :math:`\varsigma` is given by

.. math::

    \varsigma(f_1,\dots,f_{n-1}) = 
    \left(\sum_{i=1}^{n-1} \varphi_i(f_i)\right) - 
    \varphi_{n}%(f_k)
    \left({\textstyle\sum}_{j=1}^{n-1} f_j \right),

defining the phases :math:`\varphi_i` via :math:`\hat x_i(f) = |\hat x_i(f)|e^{i\varphi_i(f)}`.