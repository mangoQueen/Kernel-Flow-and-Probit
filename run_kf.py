import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import perf_counter
from kernelflow import theta_newt
import autograd.numpy as np
import sys
import argparse

if __name__ == "__main__":
    #Set up Data
    # Following example in [3]:

    N_each = 50 # Number of points in each cluster
    n_cluster = 3 # Number of clusters
    N = N_each*n_cluster # Total number of points

    mu, sigma = 0, 0.1 # mean and standard deviation
    xs = np.random.normal(mu, sigma, N)
    ys = np.random.normal(mu, sigma, N)
    zs = np.random.normal(mu, sigma, N)

    # Cluster centers: (1,0,0) (0,1,0) (0,0,1)
    xs[:N_each] += 1; ys[N_each:2*N_each] += 1; zs[2*N_each:3*N_each] += 1
    Data = np.array([xs,ys,zs]).T

    u_dagger = np.append(np.ones(2*N_each), -1*np.ones(N_each))
    Z_prime = np.array([0,50,100]) # Indices of labels
    y = np.sign(u_dagger[Z_prime]) # One label is observed within each cluster

    # Display plot

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.scatter(xs[:2*N_each], ys[:2*N_each], zs[:2*N_each], 'r')
    # ax.scatter(xs[2*N_each:3*N_each], ys[2*N_each:3*N_each], zs[2*N_each:3*N_each], 'b')
    # ax.set_title('Labeled Data with ' + str(N) +' points')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z');
    # plt.show()

    parser = argparse.ArgumentParser()

    #g, alpha, tau, eps, rval = theta
    parser.add_argument("-g", help= "std defined for psi function")
    parser.add_argument("-alpha", help= "alpha for Covariance matrix")
    parser.add_argument("-tau", help= "tau for Covariance matrix ")
    parser.add_argument("-eps", help= "eps for kernel function")
    parser.add_argument("-r", help="cutoff for kernel ")
    parser.add_argument("-l", help="learning rate for gradient descent")
    parser.add_argument("-tol", help="tolerance for gradient descent ")
    parser.add_argument("-m", help="maximu iteration for gradient descent ")

    g = 0.5; alpha = 1; tau = 1; eps = 0.15; rval = 0.25
    learning_rate = 1e-6; tol=1e-05; maxiter=10
    args = parser.parse_args()
    if args.g:
        g = float(args.g)
    if args.alpha:
        alpha = float(args.alpha)
    if args.tau:
        tau = float(args.tau)
    if args.eps:
        eps = float(args.eps)
    if args.r:
        rval = float(args.r)
    if args.l:
        learning_rate = float(args.l)
    if args.tol:
        tol = float(args.tol)
    if args.m:
        maxiter = float(args.m)

    theta_0 = np.array([g, alpha, tau, eps, rval])
    theta, it = theta_newt(Data, N, Z_prime, y, theta_0, learning_rate, tol, maxiter)
    print("Number of Iterations: " + str(it))
    print("Theta: " + str(theta))
