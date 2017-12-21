### Notes
Details about the implementation of each routine are provided directly within the code through verbose comments. They include also lists and explanations of the arguments needed, returns and options. Here, instead, a description of the different routines is provided:


#### `confidence_band` 
creates confidence bands of a model optimized by a curve fit regression using Least Square (LS) method. It is able to handle with different cases: Linear or Ordinary Least Square (OLS), Non-Linear Least Square (NLLS), unknown and known variances (the latter very dear to those who works with experimental data) and then homoscedasticity or heteroskedasticity of the data. Some technical and statistical details:

  * **Why** The idea behind the creation of this routine is to permit to handle with different cases of fit of the data. Searching on the web it is possible to find a lot of documents and tutorials explaining how to create confidence or prediction intervals (not only in Python), but most of the times they focus only on some very limited (but widely requested) cases such as simple OLS regression or non-experimental data (...in Physics we have errors! Damn!). In other cases there is a way to solve the problem but it requires to use packages or libraries you are not familiar with or you cannot use on your machines. This routine tries to solve these two aspects (at least for the author).

  * **How (code)** It is a sort of wrapper of the famous Scipy function [`scipy.optimize.curve_fit`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html) and it is based on it: shared positional or keyword arguments follow the roles and the definitions used in Scipy so as to make the routine quickly usable for those already familiar with `curve_fit` itself. Needs Scipy v0.14.0 or later versions (needs the argument `absolute_sigma`). Here the prototype and description:

    ```python

    def confidence_band(model, xdata, ydata, confidence_level=0.6827,
                    xvals=None, prediction=False, bootstrap=False,
                    num_boots=1000, full_output=False, **kwargs):

    """Compute confidence or prediction bands of a model optimized by a
    curve fit regression with Least Square method.

    Parameters:
    -----------
    
    model : callable
              The model function f(x,*p) taking the independent
              variable as first argument and the fit parameters as
              separate remaining arguments.

    xdata : scalar or array-like
              Measured independent variable values.

    ydata : scalar or array-like
              Dependent data.

    confidence_level : number, optional
              Desired confidence level used in the definition of the
              bands. It must be a number in the range (0,1).  Default
              value is 0.6827, corresponding to one 'gaussian' sigma.
              Other common values are 0.9545 ('two sigmas') and 0.9973
              ('three sigmas').

    xvals : scalar or array-like, optional
              Value(s) of the indipendent variable where the band is
              computed. If not given, xdata is used as default. This
              value is also overrided by xdata if prediction=True and
              if absolute_sigma is passed as keyword argument to
              curve_fit (see below).

    prediction : bool, optional 
              If True, compute prediction intervals instead of
              confidence band.

    bootstrap : bool, optional
              If True, intervals are computed by using the boostrap
              method. This should be the preferred choice in the case
              of a NLLS regression.

    num_boots : int, optional
              Number of bootstrap resamples used if bootstrap method
              is selected. Default is 1000 if absolute_sigma is True,
              otherwise it is set to the minimum between 1000 and N^N,
              where N is number of data points.

    full_output : bool, optional
              If False, upper and lower bounds of the band are
              returned. Otherwise, also central predicted response,
              optimized parameters and covariance matrix of the
              regressor are inserted. Default is False.

    kwargs              
              Keyword arguments passed to curve_fit. In particular,
              the values of keyword argument 'absolute_sigma' and
              'sigma' are crucial and uses in the calculation of the
              bands.

    Returns: 
    --------

    upper : scalar or array like
              Upper bound on the confidence (or prediction) band.

    lower : scalar or array like
              Lower bound on the confidence (or prediction) band.

    central : scalar or array like, optional
              Mean predicted response curve, the model with optimized
              values of the regression parameter

    popt : scalar or array like, optional
              Optimized values of the regression parameters.

    pcov : 2d array
              The estimated covariance of popt. The diagonals provide
              the variance of the parameter estimate.


    Notes:
    ------

    I) The variance of the predicted values by the regression is
    defined by using the (approximate) jacobian matrix of the model
    with respect to the fit parameter. This method is robust in the
    case of Ordinary Least Square regression (OLS), i.e. when the
    model is linear in the parameter, but it is also used in the
    non-linear case (NLLS) despite a bootstrap procedure should be
    preferred (see below) 

    II) Approximate bootstrap confidence intervals requires different
    methods in the case of absolute_sigma or not. If true errors are
    given, then the bootstrap is over the measures, assuming normal
    distribution around ydata with spread given by sigma. If
    absolute_sigma is False, residuals are bootstrapped. In this case,
    the number of total possible resamplings with replacement is N^N,
    where N is the number of data points.  It is left to the user to
    decide which value of num_boots is good.

    III) Prediction interval can be computed only on original data 
    'xdata' if absolute_sigma is True.

    """

    ```

  * **How (statistics)** Confidence bands are computed directly using the observation/regression matrix _X_ which appears in the fit model _Y=Xb_ where _Y_ is the dependent variable vector and _b_ the vector of regression parameters. It is used to estimate the variance of the mean predicted response _Y*_ which is given by _s^2* = X w^2 X^t_ where w^2 is the covariance matrix of the fit parameters. The correct score that multiplies the variance depends on the case (z-score for known variances, t-score for unknown variance with homoscedasticity assumption, using the Mean Square Error (MSE) to estimate the unique variance) and on the desired confidence level (CL). Obviously, this computation coincides with the formulas easy to find on the web about some simpler cases such as simple linear regression. More in general, this method is robust in the case of OLS but it is also used in the case of NLLS by approximating numerically the Jacobian matrix near the optimized value of the fit parameters. In this case, however, the bootstrap method would be preferred (resampling the residuals in the case of unknown variances or sampling randomly around measures if variances are given). Prediction bands can be also computed from data. Finally, some important remarks about use and limitations. The routine can handle correlated data (if covariance matrix of the observation is given) but robustness may be threatened (also in the OLS case). The same may happens if bounds are provided to the observations in order to solve a constrained problem.

  * **What** Basically, this function returns two arrays containing the upper and lower bounds of the confidence (or prediction) band around the curve defined by the mean predicted response. In its lightest use, it requires two arrays containing, respectively, the _x_ independent variable and the corresponding values of the dependent variable _y_, and the model used into the fit. Then, one can set the desired confidence level (CL), to define instead a prediction interval and/or in which range the band itself must be defined. Every other keyword parameter such as variances/errors (sigma), known variances (absolute_sigma) and so on are exactly the same required and passed to the `curve_fit` function of Scipy. See the examples!



#### `chi2_gof_test` 
performs a ChiSquare Goodness-of-Fit test to check if a model is actually able to describe the observed data.

  * **How (code)** Here the prototype of the function:

    ```python

    """Performs the Chi Square Goodness of Fit test on a model
    with parameters optimized through a least square curve fit
    procedure in 1d.

    Parameters:
    -----------
    
    model: callable
             The model function f(x, *p) which takes the independent
             variable as first argument and the fit parameters as
             separate remaining arguments.

    xdata : array-like
             Measured independent variable values.

    ydata : array-like
             Dependent data.

    popt : scalar or array-like
             Optimized values of the fit parameters as resulted from
             the curve fit procedure.

    sigma : scalar or array-like, optional
             Determines the uncertainty in 'ydata'. 
   
    counts : bool, optional
             If True, ydata are considered frequencies or counts so that
             sigma is substituted by the expected frequencies.

    full_output : bool, optional
             If True, the Mean Squared Error (MSE) is returned together with 
             the Sum of Squared Error (SSE), the degrees of freedom and the
             p-value resulted from the chi2 distribution. Default is False,
             with only the first argument returned.
    
    Returns:
    --------

    MSE : scalar
             The Mean Squared Error (MSE) of the data on the test model,
             which under assumptions is a reduced chi2 variable.

    SSE : scalar, optional
             The Sum of Squared Errors of the data on the test model.

    ndof : int, optional
             Number of degrees of freedom.

    pvalue: scalar, optional
             The obtained p-value.

    Notes:
    ------
    In this version data and variances are assumed to be uncorrelated.

    """	
 
    ```

  * **How (statistics)** The function computes the residuals of the observations with respect to the model and uses its sum to obtain the result of the test. Mean Square Error (MSE) is returned by default, but also other values (such as the p-value obtained according to the ChiSquare distribution) can be printed. The routine can handle automatically (with the keyword argument `counts=True`) the case frequency observations, i.e. by setting the variances to the square root of the expected frequencies (given by the model). Notice that if data are not counts and `sigma` is not given, these are set to one.

  * **What** By default, this routine returns the Mean Square Error (MSE), or Mean Sum of Square Residuals (MSSR) of the data with respect to the model. It needs the model, the array of the independent variable and that of the dependent one. Also variances can be passed, or a flag that tells the routine to handle with frequencies/counts. A full output can be asked by using the corresponding keyword argument, so that to return also the Sum of Square Errors (SSE) or Sum of Square Residuals (SSR), the degrees of freedom and the p-value according to the ChiSquare distribution. See the examples!



### References
Here a list of some useful references (especially about statistics) I used to write this code. The first two are related to statistical theory and methods, the second pair describes the Bootstrap method and the last one is a link to the documentation of the Scipy modules used in this project.

1) Del Prete T. (2000), "Methods of Statistical Data Analysis in High Energy Physics"	
2) Wikipedia contributors. "Ordinary least squares." Wikipedia, The Free Encyclopedia.
3) Efron, B. (1981). "Nonparametric estimates of standard error: The jackknife, the bootstrap and other methods"
4) Fox, J. (2002). "Bootstrapping regression models"
5) [scipy.optimize.curve_fit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html), [scipy.optimize.least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html) which are heavily used in this module



