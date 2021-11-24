import torch
from torch.autograd import Variable
import utils

# -----------Newton's Method----------------------------------

# Newtons method function
def newton(f, x0, tol=1e-08, maxiter=50):
    '''
    f - input function
    x0 - initialization
    tol - tolerance for step size
    '''
    x = Variable(x0, requires_grad = True)
#     for _ in range(maxiter):
#         val = f(x)
#         val.backward()
#         step = (val/x.grad)
#         x.data -= step.data
#         a = torch.linalg.norm(step)
#         if a < tol:
#             break
# #         if torch.linalg.norm(step) < tol:
# #             break
#         x.grad.data.zero_()
    for it in range(maxiter):
        hess = torch.autograd.functional.hessian(f, x)
        grad = torch.autograd.grad(f(x), x)
        step = torch.linalg.solve(hess, -grad[0])
        x.data += step
        if torch.linalg.norm(step) < tol:
            break

    x.requires_grad = False
    return x.data

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
    for j,Z_j in enumerate(Z_p):
        u_j = torch.where(N_lst == Z_j)
        S = S - torch.log(utils.cap_psi(y[Z_j]*u[u_j],g))
    return S


# Returns u* using newton's method
def u_ast_Newt(X, N_lst, g, alpha, tau, eps, rval, y, Z_p, x_0 = True):
    '''
    X - N*3 vector containing (x, y, z) of all data (assume same size)
    N_lst - indices of X to use (Size: N) (ex. for full N, N_lst = np.arange(N))
            (if just given size N, change it to list)
    eps - gives perturbed kernel if eps != 0
    tau, alpha - parameters for covariance
    rval - threshold in kernel function
    '''
    # adjusted for autograd: floats saving parameters
    if not torch.is_tensor(N_lst):
        N_lst = torch.arange(N_lst)

    C_inv = utils.Cov_inv(X, N_lst, eps, alpha, tau, rval)
    def probit_min(u):
        # Minimizer u for problem defined in [3]-(3)
        final = 0.5*torch.dot(u, torch.matmul(C_inv.float(),u)) + misfit(u, N_lst, y, Z_p, g)
        return final
    if x_0: x_0 = torch.zeros(*N_lst.size())
    return newton(probit_min, x_0)
