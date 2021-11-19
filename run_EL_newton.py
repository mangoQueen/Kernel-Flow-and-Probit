import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import autograd.numpy as np
from newton import u_ast_Newt
from EL import u_ast_EL
from time import perf_counter


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

    # Initial Parameters
    eps = 0.15
    rval = 0.25 #threshold for kernel
    tau = 1
    alpha = 1
    g = 0.5 #Noise standard deviation
    n_eig = 20 #(62) truncation

    time_start = perf_counter()
    u_ast = u_ast_EL(Data, N, g, y, Z_prime, alpha, tau, eps, rval)
    print("Runtime for EL: " + str(perf_counter() - time_start))
    pred_error =  (sum(abs(np.sign(u_ast) - np.sign(u_dagger)))/(N)*100)
    print("Error for EL: " + str(pred_error))


    time_start = perf_counter()
    u_ast_newt = u_ast_Newt(Data, N, g, y, Z_prime, alpha, tau, eps, rval)
    print("Runtime for Newton's: " + str(perf_counter() - time_start))
    pred_error =  (sum(abs(np.sign(u_ast_newt) - np.sign(u_dagger)))/(N)*100)
    print("Error for Newton's: " + str(pred_error))
