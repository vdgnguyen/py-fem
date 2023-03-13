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

def oneLineSearch_BiSection(funcObj, x0, s, g0,
                            h = 1.5e-2,
                            Nitermax=20, 
                            tol = 1e-6, 
                            absTol = 1e-12):
    """
    solving an equation by bi-section method
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
    h: float
        Initial range.
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
    xsol: np.array
        New location.
    g: np.array
        New gradient.
    alpha : double
        Search parameter in x = x + alpha*s.
    success: boolean
        True if the line-search is successful

    """
    alpha0 = 0
    alpha1 = h    
    
    phi0=np.dot(s,g0)
    phiRef = np.abs(phi0)
    phi1 =0.
    x = np.copy(x0)
    g= None
    success=True
    # find range
    itphi = 0
    print("start finding range: ...")
    while True:
        itphi += 1
        if itphi > 1000:
            print("cannot found range")
            success = False
            phi0 = np.nan
            phi1 = np.nan
            break
        alpha0=(itphi-1)*h
        alpha1=itphi*h
        if itphi> 1:
            g0 = funcObj.gradFunc(x0+alpha0*s)
            phi0 = np.dot(s,g0)
            
        g1 = funcObj.gradFunc(x0+alpha1*s)
        phi1 = np.dot(s,g1)
        if (phi0 * phi1 <=0) or (np.abs(phi0) < absTol) or (np.abs(phi1)< absTol):
            # found range
            break
        if np.isnan(phi0) or np.isnan(phi1):
            break
    print(f"done finding range: [{alpha0:.6e} {alpha1:.6e}]")
    if np.abs(phi0) < absTol:
        print(f"result found on the left bound phi0={phi0}")
        success=True
        alpha = alpha0
        x += alpha*s
        g = g0
    elif np.abs(phi1) < absTol:
        print(f"result found on the right bound phi1={phi1}")
        success=True
        alpha = alpha1
        x += alpha*s
        g = g1
    elif np.isnan(phi0) or np.isnan(phi1):
        success=False
        alpha = 0.
        x = None
        g = None
    else:
        iteIndex=0
        while True:
            iteIndex += 1
            alpha = 0.5*(alpha0 + alpha1)
            x = x0+alpha*s
            g = funcObj.gradFunc(x)
            phi = np.dot(s,g)
            print(f"ITER {iteIndex}: alpha={alpha} phi/phiRef= {phi/phiRef} phiRef={phiRef}")
            if np.isnan(phi):
                success=False
                alpha = 0.
                x = None
                g = None
                break
            elif np.abs(phi)< tol*np.abs(phiRef) or np.abs(phi) < absTol:
                success=True
                print("convergence")
                break
            else: 
                if phi*phi0 >= 0:
                    alpha0 = alpha
                    phi0 = phi
                if phi*phi1 >=0:
                    alpha1=alpha
                    phi1=phi
    
    return {"xsol": x, 
            "gradient": g, 
            "alpha": alpha, 
            "success": success}


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
    success = True
    while True:
        itphi+=1
        A = funcObj.hessFunc(x) 
        alpha -= np.dot(s, g)/(np.dot(s, np.matmul(A,s)))
        x = x0 + alpha*s
        g = funcObj.gradFunc(x)
        phi = np.dot(s,g)
        print(f"iter = {itphi}: phi/phi0={phi/phi0:.6e} phi0 = {phi0:.6e} alpha={alpha}")
        if np.isnan(phi):
            success=False
            break
        elif np.abs(phi) < tol*np.abs(phi0) or np.abs(phi) < absTol:
            print("convergence is achieved!!!!!!!")
            break
        elif itphi > Nitermax:
            success=False
            print("maximal number reach")
            break 
    return {"xsol": x, 
            "gradient": g, 
            "alpha": alpha, 
            "success": success}
    
def gradientDescent(x0, Niter, funcObj, tol=1e-8, absTol=1e-12, lineSearchMethod="Newton"):
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
    x = x0
    g = funcObj.gradFunc(x0)
    gnorm0 = np.linalg.norm(g, np.inf)
    s = -1*g/gnorm0 # to make it dimensionless
    history = [funcObj.func(x)]
    alphas= []
    success = True
    if gnorm0 <  absTol:
        print("convergence at the beginning")
    else:
        for ii in range(1,Niter):
            if lineSearchMethod=="Bisection":
                result= oneLineSearch_BiSection(funcObj,x,s,g)
            elif lineSearchMethod=="Newton":    
                result = oneLineSearch(funcObj,x,s,g)
                if not(result["success"]):
                    result= oneLineSearch_BiSection(funcObj,x,s,g)
            else:
                raise NotImplementedError(f"line search method = {lineSearchMethod} is not implemented")
            success= result["success"]
            if not(success):
                break
            else:
                alphas.append(result["alpha"])
                x = result["xsol"]
                g = result["gradient"]
                s = -1*g/gnorm0 # to make it dimensionless
                f = funcObj.func(x)
                history.append(f)
                gnorm = np.linalg.norm(g, np.inf)
                print(f"ITER {ii}: func = {f:.6e} gnorm/gnorm0 = {gnorm/gnorm0:.6e} gnorm0={gnorm0:.6e}")
                if np.isnan(gnorm):
                    success=False
                    break
                elif gnorm/gnorm0 < tol or gnorm < absTol:
                    print("convergence is achieved")
                    success=True
                    break
    return {"xsol": x, 
            "f": history[-1],
            "history": np.array(history),
            "alpha": np.array(alphas),
            "success": success
            }

                    
def conjugateGradient(x0, Niter, funcObj, updateMethod, tol=1e-8, absTol=1e-12,lineSearchMethod="Newton"):
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
    x = x0
    g = funcObj.gradFunc(x0)
    gnorm0 = np.linalg.norm(g, np.inf)
    s = -1*g
    history = [funcObj.func(x)]
    alphas= []
    success = True
    if gnorm0 <  absTol:
        print("convergence at the beginning")
    else:
        for ii in range(1,Niter):
            # line search
            if lineSearchMethod=="Bisection":
                result= oneLineSearch_BiSection(funcObj,x,s,g)
            elif lineSearchMethod=="Newton":    
                result = oneLineSearch(funcObj,x,s,g)
            else:
                raise NotImplementedError(f"line search method = {lineSearchMethod} is not implemented")
            success= result["success"]
            if not(success):
                break
            else:
                x = result["xsol"]
                grad = result["gradient"]
                alpha = result["alpha"]
                y = grad-g
                beta = 0.
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
                if np.isnan(gnorm):
                    success=False
                    break
                elif gnorm/gnorm0 < tol or gnorm < absTol:
                    print("convergence is achieved")
                    success=True
                    break
    
    return {"xsol": x, 
            "f": history[-1],
            "history": np.array(history),
            "alpha": np.array(alphas),
            "success": success
            }


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
    success=True
    if gnorm0 <  absTol:
        print("convergence at the beginning")
    else:
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
            try:
                allA = np.linalg.solve(Av, -1*bv)
                #allA = np.minimum(allA, 1e3)
                #allA = np.maximum(allA, -1e3)                
                print(f"N = {N}, M = {M}, alphas = {allA}")
                normAlpha = np.linalg.norm(allA,np.inf)
                if np.isnan(normAlpha):
                    success=False
                    break
            except:
                success=False
                break
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
            
            if np.isnan(gnorm) or np.isnan(snorm):
                success=False
                break
            
            if gnorm/gnorm0 < tol or gnorm < absTol:
                print("convergence is archieved")
                break

    return {"xsol": x, 
            "f": history[-1],
            "history": np.array(history),
            "alpha": np.array(alphas),
            "success": success
            }
    
if __name__ == "__main__":
    import UniaxialElastic as ue
    import matplotlib.pyplot as plt
    
    plt.close("all")

    plt.figure(1)
    
    N = 500
    """
    funcObj = ue.LinearElasticFunc(N, Umax=2.,E=np.ones(N+1)*1e3)
    correctFuncMin = funcObj.trueMinFunc()
    
    x0 = funcObj.getInitialGuess()
    Niter = int(1.5*N)
    # gradient descent
    result = gradientDescent(x0, Niter, funcObj)
    plt.plot((result["history"]-correctFuncMin)/np.abs(correctFuncMin),'.--',label="Gradient descent")
    for updateMethod in ["Hestenes and Stiefel 1952", "Fletcher and Reeves 1964", 
                          "Polak and Ribiere 1969",  #Fletcher 1987",
                          "Liu and Storey 1991", "Dai and Yuan 1999", "Hager and Zhang 2005"]:
        print(f"running updateMethod = {updateMethod}")                     
        result = conjugateGradient(x0, Niter, funcObj, updateMethod)
        print(f"updateMethod = {updateMethod}, x={result['xsol']}")
        plt.plot((result["history"]-correctFuncMin)/np.abs(correctFuncMin),'.--',label=f"CG - {updateMethod}")

    Mlist = [1, 2, 3, 10]
    for M in Mlist:
        print(f"RUNNING  M = {M}")
        result = multipleDirectionSearch(x0, Niter, funcObj, M=M)
        print(f"NEEEEEEEE M = {M}, x={result['xsol']}")
        print(f"strain {np.diff(result['xsol'])}")
        plt.plot((result["history"]-correctFuncMin)/np.abs(correctFuncMin),label=f"convergence history M = {M}")

    plt.xlabel("Iterations")
    plt.ylabel("(f-fexact)")
    plt.yscale("log")
    plt.legend()
    """
    
    plt.figure(2)

    funcObj = ue.IncompressibleMooneyRivlinFunc(N, Umax=2.,E=np.ones(N+1)*1e3)
    correctFuncMin = funcObj.trueMinFunc()
    
    x0 = funcObj.getInitialGuess()
    Niter = int(3*N)
    # gradient descent
    result = gradientDescent(x0, Niter, funcObj,lineSearchMethod="Bisection")
    plt.plot((result["history"]-correctFuncMin)/np.abs(correctFuncMin),'.--',label="Gradient descent")
    allMethods=["Hestenes and Stiefel 1952", "Fletcher and Reeves 1964", 
                          "Polak and Ribiere 1969", "Fletcher 1987",
                          "Liu and Storey 1991", "Dai and Yuan 1999", "Hager and Zhang 2005"]
    for updateMethod in allMethods: 
        print(f"running updateMethod = {updateMethod}")                     
        result = conjugateGradient(x0, Niter, funcObj, updateMethod)
        print(f"updateMethod = {updateMethod}, x={result['xsol']}")
        plt.plot((result["history"]-correctFuncMin)/np.abs(correctFuncMin),'.--',label=f"CG - {updateMethod}")

    Mlist = [1,2,3,10]
    for M in Mlist:
        print(f"RUNNING  M = {M}")
        result = multipleDirectionSearch(x0, Niter, funcObj, M=M)
        print(f"NEEEEEEEE M = {M}, x={result['xsol']}")
        print(f"strain {np.diff(result['xsol'])}")
        plt.plot((result["history"]-correctFuncMin)/np.abs(correctFuncMin),label=f"convergence history M = {M}")

    plt.xlabel("Iterations")
    plt.ylabel("Function value")
    plt.yscale("log")
    plt.legend()
    
    
    plt.show()
