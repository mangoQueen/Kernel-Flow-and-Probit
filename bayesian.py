import autograd.numpy as np
from newton import u_ast_Newt
from EL import u_ast_EL
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction

# Using bayesian optimization techniques

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

# Follows rho expression given in [2]-(6)
def optimized_param(X, N, Z_prime, y, EL = False):
    # EL (boolean):  true to get optimizer for EL /
    #                false to get optimizer for Newton's
    def rho(g, alpha, tau, eps, rval):
        '''
        theta - parameters to optimize (g, alpha, tau, eps, rval)
        X - whole data (not just half)
        N - (int) number of elements
        Z_prime - indices of labeled data
        y - labels (of Z_prime)
        '''

        N_f_i, Z_half = select_Nf(N, Z_prime)

        if EL:
            uast = u_ast_EL(X, N, g, y, Z_prime, alpha, tau, eps, rval)
            uast_tild = u_ast_EL(X, N_f_i, g, y, Z_half, alpha, tau, eps, rval)
        else:
            uast = u_ast_Newt(X, N, g, y, Z_prime, alpha, tau, eps, rval)
            uast_tild = u_ast_Newt(X, N_f_i, g, y, Z_half, alpha, tau, eps, rval)

        # Compute |uast-uast_tild|^2/|uast|^2 using L2 norm
        # loop over each valid N_f_i
        num = 0.0
        denom = 0.0
        for u_i, nf_i in enumerate(N_f_i):
            num += (uast_tild[u_i] - uast[nf_i])**2
            denom += uast[nf_i]**2
        return - num/denom

    zeros = 1e-7 #0 as minimum breaks optimizer
    pbounds = {'g': (zeros,2), 'alpha':(zeros,6), 'tau':(zeros,3), 'eps':(zeros,2), 'rval':(zeros,1)}
    optim = BayesianOptimization(
        f=rho,
        pbounds=pbounds,
        verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )
    optim.maximize(
    init_points=3,
    n_iter=10,)
    return optim
