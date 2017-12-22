# minimal needed modules and suppose curve_fit_utils to be in this directory
import numpy as np
from curve_fit_utils import confidence_band


# generate some fake data according to a model
pinit = np.ones(2)
def model(x, *p):
    return p[0]+p[1]*x**2

xdata = np.arange(10)
ydata = model(xdata, *pinit)

yerrs = np.random.normal(loc=5, size=len(ydata)) # add some random errors
ydata += np.random.normal(scale=yerrs) # smear data with random noise


# fit the model and create confidence_band at 68% CL
upper, lower, f, popt, pcov = confidence_band(model, xdata, ydata,
                                              p0=pinit, sigma=yerrs, absolute_sigma=True,
                                              full_output=True)


# get also approximate confidence band with bootstrap to compare
upper_boot, lower_boot = confidence_band(model, xdata, ydata,
                                         p0=pinit, sigma=yerrs, absolute_sigma=True,
                                         bootstrap=True)


# plot data and bands
from matplotlib import pyplot as plt    
plt.errorbar(xdata, ydata, yerr=yerrs, fmt='o', color='black', label='DATA')
plt.plot(xdata, f, '--', color='black', label='FIT')
plt.fill_between(xdata, lower, upper,
                 facecolor='green', linewidth=0.,
                 alpha=0.5, label='68% CL') # band
plt.fill_between(xdata, lower_boot, upper_boot,
                 facecolor ='red', linewidth=0.,
                 alpha=0.5, label='68% CL BOOTSTRAP') # band with bootstrap
plt.legend(loc=2)
plt.show()
    

