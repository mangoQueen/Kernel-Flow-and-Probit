import utils
import scipy.optimize
import autograd.numpy as np


# -----------Directly solving EL---------------------------------

# Array given by Fj in [3] - (12), (13)
def F_sum(N_lst, g, y, Z_p, u):
    '''
    N - size of vector
    g - gamma
    y - labels
    Z_p - Z' i.e. indices of labels
    u - vector
    '''
    N = N_lst.size
    Fj = np.zeros(N)
    for j,Z_j in enumerate(Z_p):
        u_j = np.where(N_lst == Z_j)
        basis=np.zeros(N);basis[u_j]=1.0
        Fj = Fj + y[j]*utils.psi(y[j]*u[u_j], g)/utils.cap_psi(y[j]*u[u_j], g)*basis
    return Fj


# Function needed for fsolve using EL given in [3]-(2)
def u_ast_EL(X, N_lst, g, y, Z_p, alpha, tau, eps, rval, x_0 = True):
    '''
    X - N*3 vector containing (x, y, z) of all data (assume same size)
    N_lst - indices of X to use (Size: N)
            (if just given size N, change it to list)
    eps - gives perturbed kernel if eps != 0
    tau, alpha - parameters for covariance
    rval - threshold in kernel function
    x_0 - initial u_0 to minimize (if undefined, x_0 = vector of zeros)
    '''
    if type(N_lst) == int:
        N_lst = np.arange(N_lst)
    # floats saving parameters
    val_g = g
    val_eps = eps
    val_alpha = alpha
    val_tau = tau
    val_r = rval
    # if (isinstance(g, int) or isinstance(g, float)) == False:
    #     val_g = g._value
    # if (isinstance(eps, int) or isinstance(eps, float)) == False:
    #     val_eps = eps._value
    # if (isinstance(alpha, int) or isinstance(alpha, float)) == False:
    #     val_alpha = alpha._value
    # if (isinstance(tau, int) or isinstance(tau, float)) == False:
    #     val_tau = tau._value
    # if (isinstance(rval, int) or isinstance(rval, float)) == False:
    #     val_r = rval._value

    N = N_lst.size
    if x_0: x_0 = np.zeros((N,1))
    n_eig = 20 #[3]-(62) truncation
    C = utils.Cov_truncated(X, N_lst, val_eps, val_alpha, val_tau, val_r, n_eig)
    def final(u):
    #     C = Cov(X, N, eps, alpha, tau, rval)
        return u - C@F_sum(N_lst, val_g, y, Z_p, u)
    return scipy.optimize.fsolve(final, x_0)
