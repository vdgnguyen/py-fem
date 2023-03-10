import numpy as np

class GeneralFunc:
    """general function works with the optimisation algorithms
    """
    def __init__(self,):
        pass
    
    def getInitialGuess(self):
        pass
        
    def func(self,x):
        pass

    def gradFunc(self,x):
        pass
        
    def hessFunc(self,x):
        pass
        
    def trueMinFunc(self):
        pass

def oneLineSearch_divide(funcObj, x0, s, g0, Nitermax=20, tol = 1e-6, absTol = 1e-12):
    """
    Parameters
    ----------
    funcObj : GeneralFunc
        A function of multiple variables to be minimized.
    x0 : np.array
        Starting point.
    s : np.array
        A search direction.
    g0 : np.array
        Gradient of function at x0.
    Nitermax : int, optional
        Maximum number of iterations. The default is 20.
    tol : float, optional
        Relative error tolerance. The default is 1e-6.
    absTol : float, optional
        Absolte error tolerance. The default is 1e-12.

    Raises
    ------
    RuntimeError
        The maximum number of iterations is reached.

    Returns
    -------
    x: np.array
        New location.
    g: np.array
        New gradient.
    alpha : double
        Search parameter in x = x + alpha*s.

    """
    alpha=0.
    phi0 = np.dot(s,g0)
    if phi0 >0:
        raise RuntimeError("WROOONG DIRECTION")
    if np.abs(phi0) < absTol:
        print("convergece at the begining")
        return x0, g0, alpha
    itphi = 0
    h = 0.1
    phi1 = 0
    while True:
        itphi += 1
        if itphi > 20:
            raise RuntimeError("cannot found range")
        g = funcObj.gradFunc(x0+h*s)
        phi1 = np.dot(s,g)
        if phi1 > 0:
            break
        else:
            h *=2
    
    phiRef = phi0
    alpha0 = 0
    alpha1 = h
    iteIndex=0
    while True:
        iteIndex = iteIndex+1
        alpha = 0.5*(alpha0 + alpha1)
        g = funcObj.gradFunc(x0+alpha*s)
        phi = np.dot(s,g)
        print(f"ITER {iteIndex}: alpha={alpha} phi/phi0= {phi/phiRef}")
        
        if np.abs(phi)< tol*np.abs(phiRef):
            print("convergence")
            break
        
        if alpha1 - alpha0 < tol*h:
            print("convergence")
            break
        
        if phi*phi0 > 0:
            alpha0 = alpha
            phi0 = phi
        if phi*phi1 >0:
            alpha1=alpha
            phi1=phi
        
    return x0+ alpha*s, g, alpha

def oneLineSearch(funcObj, x0, s, g0, Nitermax=20, tol = 1e-6, absTol = 1e-12):
    """
    Parameters
    ----------
    funcObj : GeneralFunc
        A function of multiple variables to be minimized.
    x0 : np.array
        Starting point.
    s : np.array
        A search direction.
    g0 : np.array
        Gradient of function at x0.
    Nitermax : int, optional
        Maximum number of iterations. The default is 20.
    tol : float, optional
        Relative error tolerance. The default is 1e-6.
    absTol : float, optional
        Absolte error tolerance. The default is 1e-12.

    Raises
    ------
    RuntimeError
        The maximum number of iterations is reached.

    Returns
    -------
    x: np.array
        New location.
    g: np.array
        New gradient.
    alpha : double
        Search parameter in x = x + alpha*s.

    """
    alpha=0.
    phi0 = np.dot(s,g0)
    if np.abs(phi0) < absTol:
        print("convergece at the begining")
        return x0, g0, alpha
    itphi = 0
    g = g0
    x = x0
    while True:
        itphi+=1
        A = funcObj.hessFunc(x) 
        alpha -= np.dot(s, g)/(np.dot(s, np.matmul(A,s)))
        x = x0 + alpha*s
        g = funcObj.gradFunc(x)
        phi = np.dot(s,g)
        print(f"iter = {itphi}: phi/phi0={phi/phi0:.6e} phi0 = {phi0:.6e} alpha={alpha}")
        if np.abs(phi) < tol*np.abs(phi0):
            print("convergence is achieved!!!!!!!")
            break
        elif itphi > Nitermax:
            raise RuntimeError("maximal number reach")
            break 
    return x, g, alpha
    
def gradientDescent(x0, Niter, funcObj, tol=1e-8, absTol=1e-12):
    """
    Gradient descent function
    Parameters
    ----------
    x0 : np.array
        Initial location.
    Niter : int
        Number of descent steps.
    funcObj : GeneralFunc
        Function to be minimised.

    Returns
    -------
    np.array
        optimised location
    np.array
        history of function during descent steps.
    np.array
        history of alphas during descent steps.
    """
    N = len(x0)
    x = np.copy(x0)
    g = funcObj.gradFunc(x0)
    gnorm0 = np.linalg.norm(g, np.inf)
    s = -1*g/gnorm0 # to make it dimensionless
    history = [funcObj.func(x)]
    alphas= []
    if gnorm0 <  absTol:
        print("convergence at the beginning")
        return x, np.array(history), np.array(alphas)
    
    for ii in range(1,Niter):
        x, g, alpha = oneLineSearch_divide(funcObj,x,s,g)
        alphas.append([alpha])
        s = -1*g/gnorm0 # to make it dimensionless
        f = funcObj.func(x)
        history.append(f)
        gnorm = np.linalg.norm(g, np.inf)
        print(f"ITER {ii}: func = {f:.6e} gnorm/gnorm0 = {gnorm/gnorm0:.6e} gnorm0={gnorm0:.6e}")
        if gnorm/gnorm0 < tol:
            break
    return x, np.array(history), np.array(alphas)
                    
def conjugateGradient(x0, Niter, funcObj, updateMethod, tol=1e-8, absTol=1e-12):
    """
    Conjugate gradient
    Parameters
    ----------
    x0 : np.array
        Initial location.
    Niter : int
        Number of descent steps.
    funcObj : GeneralFunc
        Function to be minimised.

    Returns
    -------
    np.array
        optimised location
    np.array
        history of function during descent steps.
    np.array
        history of alphas during descent steps.
    """
    x = np.copy(x0)
    g = funcObj.gradFunc(x0)
    gnorm0 = np.linalg.norm(g, np.inf)
    s = -1*g
    history = [funcObj.func(x)]
    alphas= []
    beta = 0
    if gnorm0 <  absTol:
        print("convergence at the beginning")
        return x, np.array(history), np.array(alphas)
    for ii in range(1,Niter):
        # line search
        x, grad, alpha = oneLineSearch_divide(funcObj,x,s,g,20)
        y = grad-g
        if updateMethod=="Hestenes and Stiefel 1952":
            beta = np.dot(grad,y)/np.dot(s,y)
        elif updateMethod=="Fletcher and Reeves 1964":
            beta = np.dot(grad,grad)/np.dot(g,g)
        elif updateMethod=="Polak and Ribiere 1969":
            beta = np.dot(grad,y)/np.dot(g,g)
        elif updateMethod=="Fletcher 1987":
            beta = -1.*np.dot(grad,grad)/np.dot(s,g)
        elif updateMethod =="Liu and Storey 1991":
            beta = -1*np.dot(grad,y)/np.dot(s,g)
        elif updateMethod =="Dai and Yuan 1999":
            beta = np.dot(grad,grad)/np.dot(s,y)
        elif updateMethod == "Hager and Zhang 2005":
            ff = np.dot(y,y)/np.dot(s,y)
            beta = np.dot(y-ff*s,grad)/np.dot(s,y)
        else:
            raise NotImplementedError("not implemented")
          
        alphas.append([alpha, alpha*beta])
        s = -1*grad + beta*s
        #
        f = funcObj.func(x)
        history.append(f)
        g = grad
        gnorm = np.linalg.norm(g, np.inf)
        print(f"ITER {ii}: func = {f:.6e} gnorm/gnorm0 = {gnorm/gnorm0:.6e} gnorm0={gnorm0:.6e}")
        if gnorm/gnorm0 < tol:
            break
    return x, history, np.array(alphas)


def multipleDirectionSearch(x0, Niter, funcObj, M, tol=1e-8, absTol=1e-12):
    """
    Multiple directions search technique

    Parameters
    ----------
    x0 : np.array
        Initial guess.
    Niter : int
        Number of steps.
    funcObj : GeneralFunc
        Function to be minimised.
    M : int
        Number of directions.
    tol : float, optional
        Relative error tolerance. The default is 1e-8.
    absTol : double, optional
        Absolute error tolerance. The default is 1e-12.

    Returns
    -------
    np.array
        optimised location
    np.array
        history of function during descent steps.
    np.array
        history of alphas during descent steps.

    """
    
    allSearchDirs = {}
    N = len(x0)
    x = np.copy(x0)
    g = funcObj.gradFunc(x0)
    A = funcObj.hessFunc(x)
    gnorm0 = np.linalg.norm(g, np.inf)
    allSearchDirs[0]= -1*g/gnorm0
    #print(allSearchDirs)
    history = [funcObj.func(x)]
    alphas = []
    if gnorm0 <  absTol:
        print("convergence at the beginning")
        return x, np.array(history), np.array(alphas)
    for ii in range(1, Niter):
        lastPos = max(allSearchDirs.keys())
        vv=[]
        for j in range(M):
            if lastPos-j in allSearchDirs.keys():
                if np.linalg.norm(allSearchDirs[lastPos-j],np.inf):
                    vv.append(allSearchDirs[lastPos-j])
        Mreal=len(vv)
        bv = np.zeros(Mreal)
        Av = np.zeros((Mreal,Mreal))
        for j in range(Mreal):
            bv[j] = np.dot(g,vv[j])
            for k in range(Mreal):
                Av[j,k] = np.dot(vv[j],np.matmul(A,vv[k]))
    
        #print(Av)
        allA = np.linalg.solve(Av, -1*bv)
        #allA = np.minimum(allA, 1e3)
        #allA = np.maximum(allA, -1e3)                
        print(f"N = {N}, M = {M}, alphas = {allA}")
        normAlpha = np.linalg.norm(allA,np.inf)
        
        #alphas.append(allA.tolist())
        s = allA[0]*vv[0]
        for j in range(1,Mreal):
            s += allA[j]*vv[j]
        
        x += s
        allA = np.minimum(allA, 1.)
        allA = np.maximum(allA, 0) 
        f = funcObj.func(x)
        history.append(f)
        g = funcObj.gradFunc(x)
        A = funcObj.hessFunc(x)
        gnorm = np.linalg.norm(g, np.inf)
        snorm = np.linalg.norm(s, np.inf)
        allSearchDirs[lastPos] = 1.*s/snorm
        allSearchDirs[lastPos+1] = -1.*g/gnorm
        print(f"ITER {ii}: func = {f:.6e} gnorm/gnorm0 = {gnorm/gnorm0:.6e} gnorm0={gnorm0:.6e}")
        
        if gnorm/gnorm0 < tol:
            break

    return x, history, np.array(alphas) 

    
if __name__ == "__main__":
    import UniaxialElastic as ue
    import matplotlib.pyplot as plt
    
    plt.close("all")
    plt.figure(1)
    
    N = 100
    funcObj = ue.LinearElasticFunc(N, Umax=2.,E=np.ones(N+1)*1e3)
    correctFuncMin = funcObj.trueMinFunc()
    
    x0 = funcObj.getInitialGuess()
    Niter = int(1.5*N)
    # gradient descent
    xsol, hist, alphas = gradientDescent(x0, Niter, funcObj)
    plt.plot((np.array(hist)-correctFuncMin),'.--',label="Gradient descent")
    for updateMethod in ["Hestenes and Stiefel 1952", "Fletcher and Reeves 1964", 
                          "Polak and Ribiere 1969", "Fletcher 1987",
                          "Liu and Storey 1991", "Dai and Yuan 1999", "Hager and Zhang 2005"]:
        print(f"running updateMethod = {updateMethod}")                     
        xsol, hist, alphas = conjugateGradient(x0, Niter, funcObj, updateMethod)
        print(f"updateMethod = {updateMethod}, x={xsol}")
        plt.plot((np.array(hist)-correctFuncMin),'.--',label=f"CG - {updateMethod}")

    Mlist = [1,2,10,50, N]
    for M in Mlist:
        print(f"RUNNING  M = {M}")
        xsol, hist, alphas = multipleDirectionSearch(x0, Niter, funcObj, M=M)
        print(f"NEEEEEEEE M = {M}, x={xsol}")
        print(f"strain {np.diff(xsol)}")
        plt.plot((np.array(hist)-correctFuncMin),label=f"convergence history M = {M}")

    plt.xlabel("Iterations")
    plt.ylabel("Function value")
    plt.yscale("log")
    plt.legend()
    
    
    plt.figure(2)
    
    N = 100
    funcObj = ue.IncompressibleMooneyRivlinFunc(N, Umax=2.,E=np.ones(N+1)*1e3)
    correctFuncMin = funcObj.trueMinFunc()
    
    x0 = funcObj.getInitialGuess()
    Niter = int(1.5*N)
    # gradient descent
    xsol, hist, alphas = gradientDescent(x0, Niter, funcObj)
    plt.plot((np.array(hist)-correctFuncMin),'.--',label="Gradient descent")
    for updateMethod in ["Hestenes and Stiefel 1952", "Fletcher and Reeves 1964", 
                          "Polak and Ribiere 1969", "Fletcher 1987",
                          "Liu and Storey 1991", "Dai and Yuan 1999", "Hager and Zhang 2005"]:
        print(f"running updateMethod = {updateMethod}")                     
        xsol, hist, alphas = conjugateGradient(x0, Niter, funcObj, updateMethod)
        print(f"updateMethod = {updateMethod}, x={xsol}")
        plt.plot((np.array(hist)-correctFuncMin),'.--',label=f"CG - {updateMethod}")

    Mlist = [1,2,10,50, N]
    for M in Mlist:
        print(f"RUNNING  M = {M}")
        xsol, hist, alphas = multipleDirectionSearch(x0, Niter, funcObj, M=M)
        print(f"NEEEEEEEE M = {M}, x={xsol}")
        print(f"strain {np.diff(xsol)}")
        plt.plot((np.array(hist)-correctFuncMin),label=f"convergence history M = {M}")
        
    plt.xlabel("Iterations")
    plt.ylabel("Function value")
    plt.yscale("log")
    plt.legend()
    
    
    plt.show()
