![PyBic logo](PyBic.png)

This module implements _Bicoherence Analyzer_ in Python!

## Quick start

To get rolling, place the ``pybic.py`` file in your desired directory, and try
```python
import pybic as bic
b = bic.BicAn('demo')
```

Alternatively, to analyze a time-series ``x`` sampled at ``fS``, use
```python
import pybic as bic
b = bic.BicAn(x,samprate=fS)
```

## Documentation

Check out our (in-development) [Read the Docs page](https://pybic.readthedocs.io/en/latest/) for detailed information about the ``pybic`` module and the associated ``pybic.BicAn`` class.

## Tutorials

For convenience, the following Jupyter notebooks demonstrate many of the features of PyBic:

[Guided tour](https://colab.research.google.com/drive/1GnJddGDVVIWK44B-_0Mfoe-tLKWoXFrb?usp=sharing)

[``bic.Plot()`` documentation](https://colab.research.google.com/drive/1NJmjnkhD9wWd_uYRYDWSOEatzS_5Nzm3?usp=sharing)

## Citing our work

To reference PyBic/BicAn, please refer to our article in Computer Physics Communications: [BicAn: An integrated, open-source framework for polyspectral analysis](https://doi.org/10.1016/j.cpc.2026.110097).

_Note that the preprint is available above!_

## Theory

### The bispectrum
$\mathcal{B}_{xyz}(f_1,f_2) = \langle X(f_1)Y(f_2)Z^*(f_1+f_2) \rangle$, where $x$, $y$, $z$ are time series with 
corresponding Fourier transforms $X$, $Y$, $Z$,
and $\langle ... \rangle$ denotes time averaging.

### The (squared) bicoherence spectrum
$b^2_{xyz}(f_1,f_2) = \frac{|B_{xyz}(f_1,f_2)|^2 }{ \left\langle|X(f_1)Y(f_2)|^2\right\rangle \left\langle|Z(f_1+f_2)|^2\right\rangle + \varepsilon }$,
where $\varepsilon$ is a small number meant to prevent 0/0 = ``NaN`` catastrophe.

## Contact

**Please reach out with any questions to rigzridge@gmail.com!**
