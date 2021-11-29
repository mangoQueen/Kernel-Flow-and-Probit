import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import autograd.numpy as np
from newton import u_ast_Newt
from EL import u_ast_EL
from bayesian import optimized_param

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
    Z_prime = np.array([0,50,100]) # Indices of labels (assume only know 3 indices)
    y = np.sign(u_dagger) # One label is observed within each cluster
    # Note that rest of indices aren't used even though it is initialized

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

    optimizer_EL = optimized_param(Data, N, Z_prime, y, EL = True)
    eps = optimizer_EL.max['params']['eps']
    rval = optimizer_EL.max['params']['rval']
    tau = optimizer_EL.max['params']['tau']
    alpha = optimizer_EL.max['params']['alpha']
    g = optimizer_EL.max['params']['g']
    u_ast = u_ast_EL(Data, N, g, y, Z_prime, alpha, tau, eps, rval)
    pred_error =  (sum(abs(np.sign(u_ast) - np.sign(u_dagger)))/(2*N)*100).item()
    print("Error for EL using new Parameters: " + str(round(pred_error,2)) + '%')

    optimizer_Newt = optimized_param(Data, N, Z_prime, y)
    eps = optimizer_Newt.max['params']['eps']
    rval = optimizer_Newt.max['params']['rval']
    tau = optimizer_Newt.max['params']['tau']
    alpha = optimizer_Newt.max['params']['alpha']
    g = optimizer_Newt.max['params']['g']
    u_ast_2 = u_ast_Newt(Data, N, g, y, Z_prime, alpha, tau, eps, rval)
    pred_error =  (sum(abs(np.sign(u_ast_2) - np.sign(u_dagger)))/(2*N)*100).item()
    print("Error for Newton's using new Parameters: " + str(round(pred_error,2)) + '%')
