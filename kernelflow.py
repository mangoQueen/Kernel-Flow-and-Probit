import autograd.numpy as np
import autograd
from newton import u_ast_Newt
from EL import u_ast_EL
from time import perf_counter

# Using Kernel Flow method to approximate parameters

# Return randomly selected half data indices from N that includes half of
# labeled data
# Follows from [1]
def select_Nf(N, Z_prime):
    '''
    N - (int) number of elements
    Z_prime - indices of labeled data
    '''
    N_f = int((N-Z_prime.size)/2) # Must be <= N
    half_Z = int(Z_prime.size/2+1) # half of labels

    # Randomly selected indices N_f and N_c used for X,Y,Z
    # Always need to include Z' or else cant compute
    N_f_i = np.random.choice(N, N_f, replace=False)
    Z_half= np.random.choice(Z_prime, half_Z, replace=False)
    for z in Z_half:
        if z not in N_f_i:
            N_f_i = np.append(N_f_i, z)

    N_f = N_f_i.size
    return N_f_i, Z_half

def theta_newt(X, N, Z_prime, y, theta_0, learning_rate, tol, maxiter):
    '''
    f - inputfunction
    theta_0 - initialization
    learning_rate - step size
    tol - tolerance for Gradient
    maxiter - maxmimum number of iterations
    '''
    # Follows rho expression given in [2]-(6)
    def rho(theta):
        '''
        theta - parameters to optimize (g, alpha, tau, eps, rval)
        X - whole data (not just half)
        N - (int) number of elements
        Z_prime - indices of labeled data
        y - labels (of Z_prime)
        '''
        # g = 0.15; alpha = 1.; tau = 1.;
        # eps, rval= theta

        g, alpha, tau, eps, rval = theta
        N_f_i, Z_half = select_Nf(N, Z_prime)

        uast = u_ast_Newt(X, N, g, y, Z_prime, alpha, tau, eps, rval)
        uast_tild = u_ast_Newt(X, N_f_i, g, y, Z_half, alpha, tau, eps, rval)

        # Compute |uast-uast_tild|^2/|uast|^2 using L2 norm
        # loop over each valid N_f_i
        num = 0.0
        denom = 0.0
        for u_i, nf_i in enumerate(N_f_i):
            num += (uast_tild[u_i] - uast[nf_i])**2
            denom += uast[nf_i]**2
        return num/denom

    theta = theta_0
    for it in range(maxiter):
        grad = autograd.grad(rho)
        direction = grad(theta)
        theta = theta - learning_rate*direction
        print(str(it) + " | cost: " + str(rho(theta)))
        print(str(it) + " | theta: " + str(theta))
        print(str(it) + " | direction: " + str(direction))
        if np.linalg.norm(direction) < tol:
            break
    return theta, it+1

def theta_EL(X, N, Z_prime, y, theta_0, learning_rate, tol, maxiter):
    '''
    f - inputfunction
    theta_0 - initialization
    learning_rate - step size
    tol - tolerance for Gradient
    maxiter - maxmimum number of iterations
    '''
    # Follows rho expression given in [2]-(6)
    def rho(theta):
        '''
        theta - parameters to optimize (g, alpha, tau, eps, rval)
        X - whole data (not just half)
        N - (int) number of elements
        Z_prime - indices of labeled data
        y - labels (of Z_prime)
        '''
        # should delete in actual run
        # g = 0.15; alpha = 1.; tau = 1.;
        # eps, rval= theta

        g, alpha, tau, eps, rval = theta
        N_f_i, Z_half = select_Nf(N, Z_prime)

        uast = u_ast_EL(X, N, g, y, Z_prime, alpha, tau, eps, rval)
        uast_tild = u_ast_EL(X, N_f_i, g, y, Z_half, alpha, tau, eps, rval)

        # Compute |uast-uast_tild|^2/|uast|^2 using L2 norm
        # loop over each valid N_f_i
        num = 0.0
        denom = 0.0
        for u_i, nf_i in enumerate(N_f_i):
            num += (uast_tild[u_i] - uast[nf_i])**2
            denom += uast[nf_i]**2
        return num/denom

    theta = theta_0
    for it in range(maxiter):
        grad = autograd.grad(rho)
        direction = grad(theta)
        theta = theta - learning_rate*direction
        print(str(it) + " | cost: " + str(rho(theta)))
        print(str(it) + " | theta: " + str(theta))
        print(str(it) + " | direction: " + str(direction))
        if np.linalg.norm(direction) < tol:
            break
    return theta, it+1
