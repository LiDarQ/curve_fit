# curve_fit

I just want to fit a gaussian curve,in this note,you can get an information that about how to use curve_fit,which parameter you should use etc.

**from scipy.optimize import curve_fit
scipy.optimize.curve_fit(f, xdata, ydata, p0=None, sigma=None, absolute_sigma=False, check_finite=True, bounds=(-inf, inf), method=None, jac=None, **kwargs)
Assumes: ydata = f(xdata, *params) + eps

**Parameters:

  f: callable
    The modle function f(x,...).It must take the independent variable as the first argument and the parameters to fit as remaining        arguments.
  
  xdata:array_like or object
    The independent variable where the data is measured.
  
  ydata:array_like
    The dependent data.
  
  p0:array_like,optional(初始估计参数)
    Inital guess for the parameters. If None, then the inital values will all be 1(If the number of parameters for the function can be         determined using introspection,otherwise a ValueError is raised). 
  
  sigma:None or M-length sequence or M x M array, optional(关于ydata的不确定性)
    Determines the uncertainty in ydata. If we define residuals as r = ydata - f(xdata,*popt),then the interpretation of sigma depends       on its number of dimensions:
    >A 1-d sigma should contain values of standard deviations of errors in ydata. in this case, the optimized function is chisq =           sum((r/sigma)**2)
    >A 2-d,omission

absolute_sigma:bool,optional
    if true,sigma is used in absolute sense and the estimated parameter covariance pcov reflects these absolute values
    if false,only the relative magnitudes of the sigma values matter. The returned parameter covariance matrix pcov is based on scaling
    sigma by a constant factor. this constant is set by demanding that reduced chisq for the optimal parameters popt when using the         scaled sigma equals unity. in other words,sigma is scaled to match the sample variance of the residuals after the                       fit.Mathematically,
    pcov(absolute_sigma=False) = pcov(absolute_sigma=True) * chisq(popt)/(M-N)
    
method;{'lm','trf','dogbox'},optional
    method to use for optimization.Default is 'lm' for unconstrainde problems and 'trf' if bounds are provided.The method 'lm' won't work when the number of observations if less than the number of variables,use 'trf'or'dogbox'in this case.
Returns:

popt:array(优化后的参数)
   optimal values of paremeters so that the sum of the squard residuals of f(xdata,*popt)-ydata is minimized
pcov:2d array(popt估计的方差)
    The estimated covariance of popt. The diagonals provide the variance of the parameter estimate. To compute one standard deviation       errors on the parameters use perr=np.sqrt(np.diag(pcov)).
    How the sigma parameter affect the estimated covariance depend on absolute_sigma argument ,as described above.
Raises:
    ValueError
      if either ydata of xdata contain NaNs,or if incompatiable options are used.
    RuntimeError
      if the least-squares minimization is fails.
    OptimizeWarning
      if convariance of the parameter can not be estimated.
**Notes

  With method='lm', the algorithm uses the Levenberg-Marquardt algorithm through leastsq. Note that this algorithm can only deal with     unconstrained problems.

  Box constraints can be handled by methods ‘trf’ and ‘dogbox’. Refer to the docstring of least_squares for more information.    
  
**Examples

    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    def func(x,a,b,c)
      return(a * np.exp(-b * x) + c)
    xdata = np.linspace(0, 4, 50)
    y = func(xdata, 2.5, 1.3, 0.5)
    np.random.seed(1729)
    y_noise = 0.2 * np.random.normal(size=xdata.size)
    ydata = y + y_noise
    plt.plot(xdata, ydata, 'b-', label='data')
    popt, pcov = curve_fit(func, xdata, ydata)
    plt.plot(xdata, func(xdata, *popt), 'r-',label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

Constrain the optimization to the region of 0 <= a <= 3, 0 <= b <= 1 and 0 <= c <= 0.5:

    popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))
    plt.plot(xdata, func(xdata, *popt), 'g--',label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()#图例
    plt.show()
