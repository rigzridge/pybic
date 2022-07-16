# pybic
Python-based bicoherence analysis

## The bispectrum
$\mathcal{B}_{xyz}(f_1,f_2) = \langle X(f_1)Y(f_2)Z^*(f_1+f_2) \rangle$, where $x$, $y$, $z$ are time series with 
corresponding Fourier transforms $X$, $Y$, $Z$,
and $\langle ... \rangle$ denotes averaging

## The (squared) bicoherence spectrum
$b^2_{xyz}(f_1,f_2) = \frac{|B_{xyz}(f_1,f_2)|^2 }{ \left\langle|X(f_1)Y(f_2)|^2\right\rangle \left\langle|Z(f_1+f_2)|^2\right\rangle + \varepsilon }$,
where $\varepsilon$ is a small number meant to prevent 0/0 = ``NaN`` catastrophe.
