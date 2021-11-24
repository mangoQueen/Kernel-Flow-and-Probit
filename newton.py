import autograd.numpy as np
import autograd
import utils

# -----------Newton's Method----------------------------------

# Newtons method function
def newton(f, x0, tol=10e-08, maxiter=50):
    '''
    f - input function
    x0 - initialization
    tol - tolerance for step size
    '''
    g = autograd.grad(f)
    h = autograd.hessian(f)

    x = x0
    for _ in range(maxiter):
        step = np.linalg.solve(h(x), -g(x))
        x = x + step
        if np.linalg.norm(step) < tol:
            break

    return x

def misfit(u, N_lst, y, Z_p, g):
    '''
    Misfit function defined in [3]-(45)
    u - vector to find
    N_lst - list of indices used
    y - labels
    Z_p - Z' i.e. indices of labels
    u - vector
    '''
    S = 0.0
    for Z_j in Z_p:
        u_j = np.where(N_lst == Z_j)
        S = S - np.log(utils.cap_psi(y[Z_j]*u[u_j][0],g))
    return S


# Returns u* using newton's method
def u_ast_Newt(X, N_lst, g, y, Z_p, alpha, tau, eps, rval, x_0 = True):
    '''
    X - N*3 vector containing (x, y, z) of all data (assume same size)
    N_lst - indices of X to use (Size: N) (ex. for full N, N_lst = np.arange(N))
            (if just given size N, change it to list)
    eps - gives perturbed kernel if eps != 0
    tau, alpha - parameters for covariance
    rval - threshold in kernel function
    '''
    # adjusted for autograd: floats saving parameters
    val_g = g
    val_eps = eps
    val_alpha = alpha
    val_tau = tau
    val_r = rval
    if isinstance(g, autograd.numpy.numpy_boxes.ArrayBox):
        val_g = g._value
    if isinstance(eps, autograd.numpy.numpy_boxes.ArrayBox):
        val_eps = eps._value
    if isinstance(alpha, autograd.numpy.numpy_boxes.ArrayBox):
        val_alpha = alpha._value
    if isinstance(tau, autograd.numpy.numpy_boxes.ArrayBox):
        val_tau = tau._value
    if isinstance(rval, autograd.numpy.numpy_boxes.ArrayBox):
        val_r = rval._value

    if type(N_lst) == int:
        N_lst = np.arange(N_lst)

    C_inv = utils.Cov_inv(X, N_lst, val_eps, val_alpha, val_tau, val_r)
    def probit_min(u):
        # Minimizer u for problem defined in [3]-(3)
        return 1/2*np.dot(u, np.matmul(C_inv,u)) + misfit(u, N_lst, y, Z_p, g)
    if x_0: x_0 = np.zeros(N_lst.size)
    return newton(probit_min, x_0)
