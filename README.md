# Curve Fit Utils

`curve_fit_utils` is a Python module containing some simple but useful tools for curve fitting and regression.

### Overview
The aim is to provide a readable and reusable code made from scratch and based on Numpy and Scipy modules. The module is essentially a collection of routines that I use often during my work, so I simply decided to organize the code and make it reusable. The idea is to take care of portability (often I need to use and call these routines quickly on different environments or devices) and adaptability to other codes. At present, this module contains

1) A routine that allows for the ==computation of confidence (or prediction) bands of a fit model==.
2) Tool that implements the ==Chi-square goodness-of-fit test==.

A somehow verbose and detailed description of the modules (both about the implementation and the statistical part) can be found below in this document. However, I suggest to go directly to the examples below and then only later to read all the (probably boring) details.

### Usage
Here I report some simple examples of use of the routines contained in `curve_fit_utils`. Let's suppose to have some data and a model we want to fit

```python
def model (x, *p):
    return p[0]+p[1]*x**2
    #or also NL models such as p[0]*np.exp(-p[1]*x)/x and so on
```

if we are interest in constructing a confidence band around the fit curve, we obtain them by using the function `confidence_band` with the desired (say 95%) CL

```python
from curve_fit_utils import confidence_band
# xdata and ydata are the independent and dependent variable arrays
upper, lower = confidence_band(model, xdata, ydata, confidence_level=0.95)
```

where `upper` and `lower` are Numpy arrays containing the bounds of the band. If needed one can use this routine as a sort of wrapper of `scipy.optimize.curve_fit` and get the full output

```python
upper, lower, central, popt, pcov = confidence_band(model, xdata, ydata, 
                                                    confidence_level=0.95,
                                                    full_output=True)
```

obtaining also the fit curve `central` of the mean predicted response (just the optimized model), the optimized values of the parameters `popt` and the estimated covariance matrix `pcov` of them, exactly as `curve_fit` does. Finally we can plot all the results (here I use `matplotlib.pyplot`)

```python
from matplotlib import pyplot as plt
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(xdata, central, 'r--', label='fit')
plt.fill_between(xdata, lower, upper, facecolor='gray', alpha='0.3') 
plt.legend()
plt.plot()
```

so that the confidence band is plotted in trasparency together with the fit curve and the data. Now we change our mind and make new requests: we want to define prediction bands instead of confidence bands, to define them on a different range and to use bootstrap method in the computation. Then

```python 
x = np.linspace(min(xdata), max(xdata) #create a more dense range of points
upper_pred, lower_pred = confidence_band(model, xdata, ydata, 
                                         xvals=x, predition=True, bootstrap=True)
```

and now the bounds can be plotted against the new range `x` defined above. Finally, let's suppose now to be in the known variance case, i.e. our data are measures affected by an error. We have to tell it to the routine by passing the correct keyword arguments which are the same used in Scipy in `curve_fit`

```python
#yerrs is the array with the errors of ydata
upper, lower = confidence_band(model, xdata, ydata, sigma=yerrs, absolute_sigma=True)
```

which will give us confidence bands when variances are known (and they are not only weights). In this case the CL is 68% by default. Finally, we want also to test the model, i.e. we ask if it is statistically acceptable to describe our data. To do so we use the `chi2_gof_test` function

```python
from curve_fit_utils import chis2_gof_test
#suppose optimized parameter array 'popt' is given by previous computation
MSE, SSE, ndof, pvalue = chi2_gof_test(model, xdata, ydata, popt, 
                                       sigma=yerrs, full_output=True
```

then we can look at the Mean Square Error `MSE` which is defined as the Sum of Square Errors `SSE` divided by the degrees of freedom `ndof`. If we believe our data to be normally distributed, then SSE would be distributed as ChiSquare variable (in the same way, MSE as a reduced ChiSquare) and our hypothesis about the model can be checked by looking at the value of `pvalue` which is the p-value.


### Description
Details about the implementation of each routine are provided directly within the code through verbose comments. They include also lists and explanations of the arguments needed, returns and options. Here, instead, a description of the different routines is provided:

* `confidence_band` creates confidence bands of a model optimized by a curve fit regression using Least Square (LS) method. It is able to handle with different cases: Linear or Ordinary Least Square (OLS), Non-Linear Least Square (NLLS), unknown and known variances (the latter very dear to those who works with experimental data) and then homoscedasticity or heteroskedasticity of the data. Some technical and statistical details:

  * **Why** The idea behind the creation of this routine is to permit to handle with different cases of fit of the data. Searching on the web it is possible to find a lot of documents and tutorials explaining how to create confidence or prediction intervals (not only in Python), but most of the times they focus only on some very limited (but widely requested) cases such as simple OLS regression or non-experimental data (...in Physics we have errors! Damn!). In other cases there is a way to solve the problem but it requires to use packages or libraries you are not familiar with or you cannot use on your machines. This routine tries to solve these two aspects (at least for the author).

  * **How (code)** It is a sort of wrapper of the famous Scipy function [`scipy.optimize.curve_fit`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html) and it is based on it: shared positional or keyword arguments follow the roles and the definitions used in Scipy so as to make the routine quickly usable for those already familiar with `curve_fit` itself. Needs Scipy v0.14.0 or later versions (needs the argument `absolute_sigma`).

  * **How (statistics)** Confidence bands are computed directly using the observation/regression matrix _X_ which appears in the fit model _Y=Xb_ where _Y_ is the dependent variable vector and _b_ the vector of regression parameters. It is used to estimate the variance of the mean predicted response _Y*_ which is given by _s^2* = X w^2 X^t_ where w^2 is the covariance matrix of the fit parameters. The correct score that multipies the variance depends on the case (z-score for known variances, t-score for unknown variance with homoscedasticity assumption, using the Mean Square Error (MSE) to estimate the unique variance) and on the desired confidence level (CL). Obviously, this computation coincides with the formulas easy to find on the web about some simpler cases such as simple linear regression. More in general, this method is robust in the case of OLS but it is also used in the case of NLLS by approximating numerically the Jacobian matrix near the optimized value of the fit parameters. In this case, however, a bootstrap method would be preferred (resampling the residuals in the case of unknown variances or sampling randomly around measures if variances are given). Prediction bands can be also computed from data. Finally, some important remarks about use and limitations. The routine can handle correlated data (if covariance matrix of the observation is given) but robustness may be threatened (also in the OLS case). The same may happens if bounds are provided to the observations in order to solve a constrained problem.

  * **What** Basically, this function returns two arrays containing the upper and lower bounds of the confidence (or prediction) band around the curve defined by the mean predicted response. In its lightest use, it requires two arrays containing, respectively, the _x_ independent variable and the corresponding values of the dependent variable _y_, and the model used into the fit. Then, one can set the desired confidence level (CL), to define instead a prediction interval and/or in which range the band itself must be defined. Every other keyword parameter such as variances/errors (sigma), known variances (absolute_sigma) and so on are exactly the same required and passed to the `curve_fit` function of Scipy. See the examples!

* `chi2_gof_test` performs a ChiSquare Goodness-of-Fit test to check if a model is actually able to describe the observed data.

  * **How (statistics)** The function computes the residuals of the observations with respect to the model and uses its sum to obtain the result of the test. Mean Square Error (MSE) is returned by default, but also other values (such as the p-value obtained according to the ChiSquare distribution) can be printed. The routine can handle automatically (with the keyword argument `counts=True`) the case frequency observations, i.e. by setting the variances to the square root of the expected frequencies (given by the model). Notice that if data are not counts and `sigma` is not given, these are set to one.

  * **What** By default, this routine returns the Mean Square Error (MSE), or Mean Sum of Square Residuals (MSSR) of the data with respect to the model. It needs the model, the array of the independent variable and that of the dependent one. Also variances can be passed, or a flag that tells the routine to handle with frequencies/counts. A full output can be asked by using the corresponding keyword argument, so that to return also the Sum of Square Errors (SSE) or Sum of Square Residuals (SSR), the degrees of freedom and the p-value according to the ChiSquare distribution. See the examples!

### TODO
- Add statistical documentation
- Check for errors in this README
- Create bootstrap method in confidence_band
- Add counts method to chi2_gof_test






