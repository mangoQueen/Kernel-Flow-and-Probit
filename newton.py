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

def misfit(u, y, Z_p, g):
    '''
    Misfit function defined in [3]-(45)
    N - size of vector
    y - labels
    Z_p - Z' i.e. indices of labels
    u - vector
    '''
    S = 0
    for j,Z_j in enumerate(Z_p):
        S = S - np.log(utils.cap_psi(y[j]*u[Z_j],g))
    return S


# Returns u* using newton's method
def u_ast_Newt(X, N_lst, g, y, Z_p, alpha, tau, eps, r, x_0 = True):
    '''
    X - N*3 vector containing (x, y, z) of all data (assume same size)
    N_lst - indices of X to use (Size: N) (ex. for full N, N_lst = np.arange(N))
            (if just given size N, change it to list)
    eps - gives perturbed kernel if eps != 0
    tau, alpha - parameters for covariance
    rval - threshold in kernel function
    '''
    if type(N_lst) == int:
        N_lst = np.arange(N_lst)
    C_inv = utils.Cov_inv(X, N_lst, eps, alpha, tau, r)
    def probit_min(u):
        # Minimizer u for problem defined in [3]-(3)
        return 1/2*np.dot(u, C_inv@u) + misfit(u, y, Z_p, g)
    if x_0: x_0 = np.zeros(N_lst.size)
    return newton(probit_min, x_0)
