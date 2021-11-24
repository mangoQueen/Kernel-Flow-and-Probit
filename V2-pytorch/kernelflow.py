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
    g, alpha, tau, eps, rval = theta_0

    g = Variable(torch.tensor(g), requires_grad = True)
    alpha = Variable(torch.tensor(alpha), requires_grad = True)
    tau = Variable(torch.tensor(tau), requires_grad = True)
    eps = Variable(torch.tensor(eps), requires_grad = True)
    rval = Variable(torch.tensor(rval), requires_grad = True)

    for it in range(maxiter):
        # rho function given in [2]-(6)
        N_f_i, Z_half = select_Nf(N, Z_prime)

        uast = u_ast_Newt(X, N, g, alpha, tau, eps, rval, y, Z_prime)
        uast_tild = u_ast_Newt(X, N_f_i, g, alpha, tau, eps, rval, y, Z_half)

        # Compute |uast-uast_tild|^2/|uast|^2 using L2 norm
        # loop over each valid N_f_i
        num = 0.0
        denom = 0.0
        for u_i, nf_i in enumerate(N_f_i):
            num += (uast_tild[u_i] - uast[nf_i])**2
            denom += uast[nf_i]**2
        cost = num/denom

        print(str(it) + " | cost: " + str(cost.item()))

        cost.backward()
        with torch.no_grad():
            g -= learning_rate * g.grad
            alpha -= learning_rate * alpha.grad
            tau -= learning_rate * tau.grad
            eps -= learning_rate * eps.grad
            rval -= learning_rate * rval.grad

            g.grad = None
            alpha.grad = None
            tau.grad = None
            eps.grad = None
            rval.grad = None


        print(str(it) + " | theta: " + str(theta))
    return theta, it+1
