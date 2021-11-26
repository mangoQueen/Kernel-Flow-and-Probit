import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from newton import u_ast_NEWT, u_ast_GRAD
from EL import u_ast_EL
from time import perf_counter


if __name__ == "__main__":
    #Set up Data
    # Following example in [3]:

    N_each = 50 # Number of points in each cluster
    n_cluster = 3 # Number of clusters
    N = N_each*n_cluster # Total number of points

    xs = torch.rand(N)
    ys = torch.rand(N)
    zs = torch.rand(N)

    # Cluster centers: (1,0,0) (0,1,0) (0,0,1)
    xs[:N_each] += 1; ys[N_each:2*N_each] += 1; zs[2*N_each:3*N_each] += 1
    Data = torch.stack((xs, ys, zs), 1)

    # true labels
    u_dagger = torch.cat((torch.ones(2*N_each), -1*torch.ones(N_each)))
    Z_prime = [0,50,100] # Indices of labels
    y = torch.sign(u_dagger) # One label is observed within each cluster

    # Display plot
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(xs[:2*N_each], ys[:2*N_each], zs[:2*N_each], 'r')
    ax.scatter(xs[2*N_each:3*N_each], ys[2*N_each:3*N_each], zs[2*N_each:3*N_each], 'b')
    ax.set_title('Labeled Data with ' + str(N) +' points')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z');
    plt.show()

    # Initial Parameters
    eps = 0.15
    rval = 0.25 #threshold for kernel
    tau = 1
    alpha = 1
    g = 0.5 #Noise standard deviation
    n_eig = 20 #(62) truncation

    time_start = perf_counter()
    u_ast = u_ast_EL(Data, N, g, alpha, tau, eps, rval, y, Z_prime)
    print("Runtime for EL: " + str(round(perf_counter() - time_start,3)) + "sec")
    pred_error =  (sum(abs(torch.sign(u_ast) - torch.sign(u_dagger)))/(2*N)*100).item()
    print("Error for EL: " + str(round(pred_error,2)) + '%')


    time_start = perf_counter()
    u_ast_2 = u_ast_NEWT(Data, N, g, alpha, tau, eps, rval, y, Z_prime)
    print("Runtime for Newton's: " + str(round(perf_counter() - time_start,3)) + "sec")
    pred_error =  (sum(abs(torch.sign(u_ast_2) - torch.sign(u_dagger)))/(2*N)*100).item()
    print("Error for Newton's: " + str(round(pred_error,2)) + '%')

    time_start = perf_counter()
    u_ast_3 = u_ast_GRAD(Data, N, g, alpha, tau, eps, rval, y, Z_prime)
    print("Runtime for EL: " + str(round(perf_counter() - time_start,3)) + "sec")
    pred_error =  (sum(abs(torch.sign(u_ast_3) - torch.sign(u_dagger)))/(2*N)*100).item()
    print("Error for EL: " + str(round(pred_error,2)) + '%')
