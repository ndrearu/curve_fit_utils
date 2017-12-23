import numpy as np
from scipy.optimize import curve_fit, approx_fprime
from scipy.stats import norm, t, chi2
import warnings


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

    # check if bounds are provided
    bounds = kwargs.get('bounds', None)
    if bounds is not None :
        warnings.warn("No guarantee bands are correct if bounds are given")
    
    # perform the fit
    popt, pcov = curve_fit(model, xdata, ydata, **kwargs)

    # some stuff needed below
    ndata = len(xdata)
    npars = len(popt)
    p0 = np.ones( npars )
    p0 = kwargs.get('p0', p0)
    sigma = np.ones( ndata )
    sigma = kwargs.get('sigma', sigma)
    absolute_sigma = kwargs.get('absolute_sigma', False)

    if not absolute_sigma :
        SSE = ydata - model(xdata, *popt)
        SSE = np.sum( ( SSE / sigma )**2 )
        MSE = SSE / (ndata - npars)


    # define x range of the band
    if xvals is not None :
        x = np.asarray(xvals)

        if prediction and absolute_sigma :
            #if known variances, prediction only on xdata
            x = np.asarray(xvals)
            warnings.warn("Prediction interval only on original data, output changed")

    else :
        x = np.asarray(xdata)


    # mean predicted response
    pr_mean = model(x, *popt)

    if not bootstrap :

        # compute jacobian around popt
        def model_p(p, z):
            return model(z, *p)

        npoints = len(x)
        jac = np.array([])
        jac_shape = (npoints, npars)

        for z in x :
            dp = approx_fprime(popt, model_p, 10e-6, z)
            jac = np.append(jac, dp)

        jac = np.reshape(jac, jac_shape)
        jac_transposed = np.transpose(jac)

        # compute predicted response variance
        # optimized way to do equivalently
        # np.diag(np.dot(jac, np.dot(pcov, jac_tranposed) )
        pr_var = np.dot(pcov, jac_transposed)
        pr_var = pr_var * jac_transposed
        pr_var = np.sum(pr_var, axis=0)

    else :

        npoints = len(x)
        # matrix of predicted values
        pr_matrix_b = np.array([])
        pr_matrix_b_shape = (num_boots, npoints)

        
        # prepare collection of resamples as a matrix
        # where the rows are the different resampling
        ydata_matrix_b = np.array([])

        if absolute_sigma :

            # this can be done quickly mimiking a multivariate
            # normal distribution centered at the observations
            ydata_cov = np.diag( sigma**2 )
            ydata_matrix_b = np.random.multivariate_normal(ydata, ydata_cov, size=num_boots)

        else :

            ypred = model(xdata, *popt)
            residuals = ydata - ypred

            # copy ypred on num_boots rows
            ydata_matrix_b = np.tile(ypred, (num_boots, 1) )

            # add pick with replacement of the residuals on all the matrix
            ydata_matrix_b += np.random.choice(residuals, size=(num_boots, ndata))

            
        # collect results from fit on all the resamplings
        for n in np.arange(num_boots) :
            ydata_b = ydata_matrix_b[n]
            popt_b, pcov_b = curve_fit(model, xdata, ydata_b, p0=popt)

            pr_mean_b = model(x, *popt_b)
            pr_matrix_b = np.append(pr_matrix_b, pr_mean_b)

        pr_matrix_b = np.reshape(pr_matrix_b, pr_matrix_b_shape)

        # compute mean
        pr_mean_b = np.mean(pr_matrix_b, axis=0)
        # compute variance
        pr_var = np.var(pr_matrix_b, axis=0, ddof=1)

        
    if not absolute_sigma :
        # estimate variance with MSE
        pr_var = MSE * pr_var

    # stat factors
    rtail = .5 + confidence_level / 2.
    if absolute_sigma :
        score = norm.ppf(rtail)
    else :
        dof = ndata - npars
        score = t.ppf(rtail, dof)


    # define intervals
    if not prediction :
        pr_band = score * np.sqrt( pr_var )
    elif absolute_sigma :
        # prediction forced on xdata
        pr_band = score * np.sqrt( sigma + pr_var )
    else :
        pr_band = score * np.sqrt( MSE + pr_var )
        
    central = pr_mean

    if bootstrap :
        # approximate bootstrap interval is around
        # the bootstrap mean, a priori not symmetric
	# around the 'standard' predicted mean
        upper = pr_mean_b + pr_band
        lower = pr_mean_b - pr_band
    else :
        upper = pr_mean + pr_band
        lower = pr_mean - pr_band

    if full_output :
        return upper, lower, central, popt, pcov
    else :
        return upper, lower
    


def chi2_gof_test(model, xdata, ydata, popt, sigma=None, counts=False, full_output=False):
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

    ndata = len(xdata)
    npars = len(popt)
    ndof = ndata - npars
    
    ypred = model(xdata, *popt)

    residuals = ypred - ydata

    if counts is True :
        sigma = np.sqrt( ypred )

    if sigma is not None :
        residuals = residuals / sigma

    SSE = np.sum( residuals ** 2 )
    MSE = SSE / ndof
    pvalue = 1. - chi2.cdf(SSE, ndof)

    if full_output :
        return MSE, SSE, ndof, pvalue
    else :
        return MSE
