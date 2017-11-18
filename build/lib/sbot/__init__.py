import numpy as np
import scipy as sp

def igd_solver(Phi,GPhi,HPhi,M,tol=1e-1,Maxiter = 50):
    '''Implicit Gradient Descent solver: takes as input a function Phi(s), its Jacobian matrix GPhi(s),
     its Hessian matrix HPhi(s) (all defined as functions) and the dimension of s which is M,
     and returns s* = argmin_s Phi(s)

     Input/type/description
     M: int, dimension of the variable we are minimizing on
     Phi: lambda (function) with input in R^M and output in R, function we are minimizing
     GPhi: lambda (function) with input in R^M and output in R^M, gradient of Phi
     HPhi: lambda (function) with input in R^M and output in R^{M \times M}, Hessian of Phi
     tol: float, max tolerance between two consecutive iteration to stop the algo
     Maxiter: int, max iterations

     Output/type/description
     s: M dimensional array of floats. Minimizer of Phi
     '''
    s = np.zeros(M)
    Id = np.eye(M)
    delta = 1.0
    Phiold = Phi(s)
    DescDir = np.linalg.solve(Id + delta*HPhi(s), GPhi(s)) # =(Id + delta*HPhi(s))^-1 * GPhi(s)
    for niter in range(Maxiter):
        #compute new point
        snew = s - delta*DescDir
        Phinew = Phi(snew)
        #if tolerance reached, break
        if abs(Phinew-Phiold)<tol:
            s = snew
            break

        if Phinew < Phiold:
            s = snew #accept update of s
            delta = 2*delta #increase delta
            Phiold = Phinew #save value for comparison
            DescDir = np.linalg.solve(Id + delta*HPhi(s) ,GPhi(s)) #compute new direction
        else:
            delta = delta / 2 #decrease delta, without accepting update of s
    return(s)

def local_sot(x,y,F,GF,tol=1e-1,Maxiter=50,solver = igd_solver):
    '''Solves the local optimal transport between two samples x and y, given a set of features
    F and their gradient GF. More details in Tabak-Kuang, Sample based OT,2017.
    This code assumes that the same features F are used to test equivalence of the samples
    and to construct the transport maps. More precisely, maps T are sought to be of the form
    T(x) = x + sum_{l=1}^M s_l F_l(x), for some coefficents s_l.
    A consequence of this restriction is that both samples x and y have to belong to the same
    euclidean space R^d (same dimension d), so that T : R^d -> R^d.
    The number of samples in x and y can be different.

    Input/type/description
    x: Array of floats (say of dimensions Nx \times d), where Nx is the number of points and d the dimension. Source sample
    y: Array of floats (say of dimensions Ny \times d), where Ny is the number of points and d (same as x) the dimension. Target sample
    F: lambda (function) with input in R^d and output in R^M. Represents the vector of features (M features)
    GF: lambda (function) with input in R^d and output in R^{M \times d}. Represents de gradient of F, each row represents the gradient (row vector) of a feature
    tol: float. Tolerance for the non-linear solver (here we use implicit gradient descent)
    Maxiter: int. Maximum number of iterations for the nonlinear solver.
    solver: function with same inputs and outputs igd_solver. Minimization solver

    Output/type description
    T: lambda (function) with input in R^d and output in R^d. Optimal transport map.
    sopt: M dimensional array. Parameters to reconstruct the map (redundant with T)
    '''

    d = len(x[0])
    Nx = len(x)
    Ny = len(y)
    M = len(F(y[0]))
    b = (1/Ny) * sum([F(y[j]) for j in range(Ny)])
    #preallocate the matrix A, where A_i =\nabla F(x_i)
    A = [GF(x[i]) for i in range(Nx)]
    #Construct Phi, GPhi and HPhi to feed in minimization solver
    G = lambda s: (1/Nx)*sum([F(x[i] + s.dot(A[i])) for i in range(Nx)])
    dG = lambda s,j,k: (1/Nx)*sum([ A[i][j,p]*GF(x[i] + s.dot(A[i]))[k,p] for p in range(d) for i in range(Nx)])
    GG = lambda s: np.array([[ dG(s,j,k) for j in range(M)] for k in range(M)])
    Phi = lambda s: 0.5 * (np.linalg.norm(G(s)-b))**2
    GPhi = lambda s: (G(s)-b).dot(GG(s))
    HPhiij = lambda s,i,j: sum( [dG(s,i,l)*dG(s,j,l) for l in range(M)])
    HPhi = lambda s: np.array([[HPhiij(s,i,j) for j in range(M)] for i in range(M)])
    #solve using nonlinear solver
    sopt = solver(Phi,GPhi,HPhi,M,tol,Maxiter)
    #construct optimal map T
    T = lambda x: x + sopt.dot(GF(x)) #maybe put GF as local here?
    return(T, sopt)


def sot(x,y,F,GF,K=10,tol=1e-1,Maxiter=50,local_ot = local_sot):
    '''Solve the optimal transport between two samples x and y, given a set of features
    F and their gradient GF, by using a local OT code to specify. More details in Tabak-Kuang,
    Sample based OT,2017.

    Input/type/description
    x: Array of floats (say of dimensions Nx \times d), where Nx is the number of points and d the dimension. Source sample
    y: Array of floats (say of dimensions Ny \times d), where Ny is the number of points and d (same as x) the dimension. Target sample
    F: lambda (function) with input in R^d and output in R^M. Represents the vector of features (M features)
    GF: lambda (function) with input in R^d and output in R^{M \times d}. Represents de gradient of F, each row represents the gradient (row vector) of a feature
    K: int, number of intermediary distribution between x and y to consider
    tol: float. Tolerance for the OT and local_ot solver
    Maxiter: int. Maximum number of iterations for OT and local_ot solver
    local_ot: solver for the local optimal transport problem, with same inputs and outputs as local_sot

    Output/type description
    x_K : Nx \times d array of floats. Image of the source sample to the target sample by the map
    s_maps : K \times M array of floats. s_maps[k] is the list of parameters that allows us to reconstruct the map between k and k+1. The composition of all these maps yields the final map.
    '''

    Nx = len(x)
    Ny = len(y)
    d = len(y[0])
    M = len(F(y[0]))

    s_maps = np.zeros((K,M))
    #Initialize intermediary distributions
    x_K = np.array([y[np.random.randint(0,Ny)] for i in range(Nx)])
    mu = []
    mu.append(x)
    for k in range(1,K):
        z = (1-k/K)*x+(k/K)*x_K
        mu.append(z)
    mu.append(y)

    for niter in range(Maxiter):
        x_Kold=x_K
        z = x
        #solve local OT'
        for k in range(1,K+1):
            Tloc, s = local_ot(z,mu[k],F,GF,tol,Maxiter)
            s_maps[k-1] = s
            z = np.array([Tloc(z[i]) for i in range(Nx)])
        x_K = z
        if (1/Nx)*np.linalg.norm(x_K-x_Kold)<tol:
            break
        #update intermediate distributions
        for k in range(1,K):
            mu[k] = (1-k/K)*x+(k/K)*x_K

    return(x_K, s_maps)


def ot_map(GF,s_maps):
    '''Reconstructs optimal transport map given the basis functions and the coefficients
    of the local maps at every steps. Returns the composition of these local maps.
    
    Input/type/description
    GF: M \times d array of lambda (functions). Contains basis functions for the map
    s_maps: K \times M array of floats. Each line represents the M coefficients to reconstruct
            the local map from step k to k+1.
    
    Output/type description
    Tnew : lambda (function) with input in R^d and output in R^d. Optimal transport map
    '''
    K, M = s_maps.shape
    Tnew = lambda x: x + s_maps[0].dot(GF(x))
    for k in range(1,K):
        Told = Tnew
        Tnew = lambda x, Toldloc = Told, kloc = k: Toldloc(x) + s_maps[kloc].dot(GF(Toldloc(x)))
    return(Tnew)
