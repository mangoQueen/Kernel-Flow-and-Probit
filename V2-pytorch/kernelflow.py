import torch
import random
from newton import u_ast_Newt

# Using Kernel Flow method to approximate parameters

# Return randomly selected half data indices from N that includes half of
# labeled data
# Follows from [1]
def select_Nf(N, Z_prime):
    '''
    N - (int) number of elements
    Z_prime - indices of labeled data
    '''
    N_f = int((N-len(Z_prime))/2) # Must be <= N
    half_Z = int(len(Z_prime)/2+1) # half of labels

    # Randomly selected indices N_f and N_c used for X,Y,Z
    # Always need to include Z' or else cant compute
    N_lst = range(N)
    N_f_i = random.sample(range(N), N_f)
    Z_half= random.sample(Z_prime, half_Z)
    for z in Z_half:
        if z not in N_f_i:
            N_f_i.append(z)

    return torch.tensor(N_f_i), Z_half

def optm_theta(X, N, Z_prime, y, theta_0, learning_rate, tol, maxiter):
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

        N_f_i, Z_half = select_Nf(N, Z_prime)

        uast = u_ast_Newt(X, N, theta, y, Z_prime)
        uast_tild = u_ast_Newt(X, N_f_i, theta, y, Z_half)

        # Compute |uast-uast_tild|^2/|uast|^2 using L2 norm
        # loop over each valid N_f_i
        num = 0.0
        denom = 0.0
        for u_i, nf_i in enumerate(N_f_i):
            num += (uast_tild[u_i] - uast[nf_i])**2
            denom += uast[nf_i]**2
        return num/denom

    theta = Variable(torch.tensor(theta_0), requires_grad = True)
    for it in range(maxiter):
        cost = rho(theta)
        print(str(it) + " | cost: " + str(cost.item()))

        cost.backward()
        with torch.no_grad():
            theta -= learning_rate * theta.grad
        theta.grad.data.zero_()

        print(str(it) + " | theta: " + str(theta))
        if np.linalg.norm(direction) < tol:
            break
    return theta, it+1
