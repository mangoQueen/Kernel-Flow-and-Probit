# Kernel-Flow-and-Probit

run_EL_newton calculates u* using newton's method for the minimizer function or directly solves using Euler-Lagrange equations. Shows computation time and error.

run_kf.py runs kernel flow to approximate theta on data set with 3 chosen clusters.  
Command line arguements for run_kf.py are:
'''
    -g        Standard deviation defined for psi function used in kappa
    -alpha    alpha for Covariance matrix
    -tau      tau for Covariance matrix
    -eps      eps for kernel function
    -r        cutoff for kernel function
    -l        learning rate for gradient descent
    -tol      tolerance for gradient descent
    -m        maximum number of iterations for gradient descent
'''
## Referenced equations from following papers

[1] H. Owhadi and G. R. Yoo. Kernel flows: From learning kernels from data into the abyss. Journal of Computational Physics, 389:22–47, 2019.

[2] B. Hamzi and H. Owhadi. Learning dynamical systems from data: a simple cross-validation perspective. CoRR, abs/2007.05074 2020.

[3] F. Hoffmann, B. Hosseini, Z. Ren, and A. Stuart. Consistency of semi-supervised learning algorithms on graphs: Probit and one-hot methods. Journal of Machine Learning Research 21, 1-55, 2019
