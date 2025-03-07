![PyBic logo](PyBic.png)

This module implements _Bicoherence Analyzer_ in Python!

To get started, place the ``pybic.py`` file in your desired directory, and try
```python
import pybic as bic
b = bic.BicAn('demo')
```

Alternatively, to analyze a time-series ``x`` sampled at ``fS``, use
```python
import pybic as bic
b = bic.BicAn(x,samprate=fS)
```
_More to come! Keep an eye out for our upcoming publication!_

[Link to tutorial notebook](https://colab.research.google.com/drive/1GnJddGDVVIWK44B-_0Mfoe-tLKWoXFrb?usp=sharing)

## Theory

### The bispectrum
$\mathcal{B}_{xyz}(f_1,f_2) = \langle X(f_1)Y(f_2)Z^*(f_1+f_2) \rangle$, where $x$, $y$, $z$ are time series with 
corresponding Fourier transforms $X$, $Y$, $Z$,
and $\langle ... \rangle$ denotes time averaging.

### The (squared) bicoherence spectrum
$b^2_{xyz}(f_1,f_2) = \frac{|B_{xyz}(f_1,f_2)|^2 }{ \left\langle|X(f_1)Y(f_2)|^2\right\rangle \left\langle|Z(f_1+f_2)|^2\right\rangle + \varepsilon }$,
where $\varepsilon$ is a small number meant to prevent 0/0 = ``NaN`` catastrophe.
