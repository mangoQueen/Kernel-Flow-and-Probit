import utils
import torch
import scipy.optimize


# -----------Directly solving EL---------------------------------

# Array given by Fj in [3] - (12), (13)
def F_sum(N_lst, g, y, Z_p, u):
    '''
    N_lst - list of indices used
    g - gamma
    y - labels
    Z_p - Z' i.e. indices of labels
    u - vector
    '''
    N = int(*N_lst.size())
    Fj = torch.zeros(N)
    for Z_j in Z_p:
        u_j = torch.where(N_lst == Z_j)
        basis=torch.zeros(N);basis[u_j]=1.0
        Fj = Fj + y[Z_j]*utils.psi(y[Z_j]*u[u_j], g)/utils.cap_psi(y[Z_j]*u[u_j], g)*basis
    return Fj


# Function needed for fsolve using EL given in [3]-(2)
def u_ast_EL(X, N_lst, g, alpha, tau, eps, rval, y, Z_p, x_0 = True):
    '''
    X - N*3 vector containing (x, y, z) of all data (assume same size)
    N_lst - indices of X to use (Size: N)
            (if just given size N, change it to list)
    eps - gives perturbed kernel if eps != 0
    tau, alpha - parameters for covariance
    rval - threshold in kernel function
    x_0 - initial u_0 to minimize (if undefined, x_0 = vector of zeros)
    '''
    if not torch.is_tensor(N_lst):
        N_lst = torch.arange(N_lst)

    N = int(*N_lst.size())
    if x_0: x_0 = torch.zeros(N)
    n_eig = 20 #[3]-(62) truncation
    C = utils.Cov_truncated(X, N_lst, eps, alpha, tau, rval, n_eig)
    def final(u):
    #     C = Cov(X, N, eps, alpha, tau, rval)

        return u-(C.float()@F_sum(N_lst, g, y, Z_p, u)).numpy()

#         return u - torch.matmul(C,F_sum(N_lst, g, y, Z_p, u))
    return torch.tensor(scipy.optimize.fsolve(final, x_0))
