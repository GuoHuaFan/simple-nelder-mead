import numpy as np


def nelder_mead(f, x0, n_iter_max=2000, xtol=1e-4, simplex_margin=1e-1):
    """Optimize f with Nelder-Mead algorithm (Downhill Simplex Method). 
    f does not have to be differentiable.
    Parameters
    ----------
    f : function
        Function to optimize.
    x0 : 1-D array
        Initial estimation of the solution.
    n_iter_max : int
        Maximum number of iterations.
    xtol : float
        Tolerance for X variations between two successive iterations.
        If ||old_X - X|| < xtol, optimization will stop.
    simplex_margin : int
        Margin to create the simplex from x0.
    Returns
    -------
    out : 1-D array
        Best final estimate.
    """
    
    N = len(x0)
    xs = np.repeat(x0.reshape(1, -1), N+1, axis=0)
    xs[1:, :] += np.eye(N) * simplex_margin
    
    old_xs = np.ones_like(xs) * np.inf
    
    n_iter = 0
    while n_iter < n_iter_max and np.sum(np.abs(xs - old_xs)) > xtol:
        n_iter += 1
        old_xs = xs[-1]
        
        evals = f(xs.T)
        
        indexes = np.argsort(evals)
        
        xs = xs[indexes]
        evals = evals[indexes]
        
        xm = np.mean(xs[:-1], axis=0)
        
        # Reflection
        xr = xm + (xm - xs[-1])
        f_xr = f(xr)
        if evals[0] <= f_xr < evals[-2]:
            xs[-1] = xr
            continue
        
        # Expansion
        if f_xr < evals[0]:
            xe = xm + 2*(xr - xm)
            if f(xe) < f_xr:
                xs[-1] = xe
            else:
                xs[-1] = xr
            continue
        
        # Contraction
        xc = xm + 0.5 * (evals[-1] - xm)
        if f(xc) < evals[-1]:
            xs[-1] = xc
            continue
        
        # Shrink
        xs[1:] = xs[0] + 0.5 * (xs[1:] - xs[0])
    
    print "n_iter : ", n_iter
    print "Last simplex variation : ", np.sum(np.abs(xs - old_xs))
    return xs[0]










if __name__ == "__main__":
        
    def f(params):
        return params[0] ** 2 + params[1] ** 2 + 42
    
    # Bad initialization
    x0 = np.array([10000., -10000.])
    
    # Maximize f
    x = nelder_mead(f, x0)
    
    # Maximum 42 is obtained at (0, 0)
    print x, f(x)
