# minimal needed modules and suppose curve_fit_utils to be in this directory
import numpy as np
from scipy.optimize import curve_fit
from curve_fit_utils import chi2_gof_test

# generate some fake data according to a model
pinit = np.ones(2)
def model(x, *p):
    return p[0]+p[1]*x**2

xdata = np.arange(10)
ydata = model(xdata, *pinit)

yerrs = np.random.normal(loc=5, size=len(ydata)) # add some random errors
ydata += np.random.normal(scale=yerrs) # smear data with random noise


# fit the model using curve_fit
popt, pcov = curve_fit(model, xdata, ydata,
                       p0=pinit, sigma=yerrs, absolute_sigma=True)


# test the model
MSE, SSE, ndof, pvalue = chi2_gof_test(model, xdata, ydata, popt
                                       sigma=yerrs, full_output=True)


# print results
print 'Reduced Chi-Square: ', MSE
print 'Chi-Square: ', SSE
print 'Degrees of freedom: ', ndof
print 'P-Value: ', pvalue

# valutate the results
if pvalue > 0 :
