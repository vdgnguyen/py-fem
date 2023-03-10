import sys
sys.path.insert(0, '../src')
import UniaxialElastic as ue
import matplotlib.pyplot as plt
import MinimisationAlgorithm as ma
import numpy as np

plt.close("all")
plt.figure(1)

N = 100
funcObj = ue.LinearElasticFunc(N, Umax=2.,E=np.ones(N+1)*1e3)
correctFuncMin = funcObj.trueMinFunc()

x0 = funcObj.getInitialGuess()
Niter = int(1.5*N)
# gradient descent
xsol, hist, alphas = ma.gradientDescent(x0, Niter, funcObj)
plt.plot((np.array(hist)-correctFuncMin)/np.abs(correctFuncMin),'.--',label="Gradient descent")
for updateMethod in ["Hestenes and Stiefel 1952", "Fletcher and Reeves 1964", 
                      "Polak and Ribiere 1969", "Fletcher 1987",
                      "Liu and Storey 1991", "Dai and Yuan 1999", "Hager and Zhang 2005"]:
    print(f"running updateMethod = {updateMethod}")                     
    xsol, hist, alphas = ma.conjugateGradient(x0, Niter, funcObj, updateMethod)
    print(f"updateMethod = {updateMethod}, x={xsol}")
    plt.plot((np.array(hist)-correctFuncMin)/np.abs(correctFuncMin),'.--',label=f"CG - {updateMethod}")

Mlist = [1,2,10,50, N]
for M in Mlist:
    print(f"RUNNING  M = {M}")
    xsol, hist, alphas = ma.multipleDirectionSearch(x0, Niter, funcObj, M=M)
    print(f"NEEEEEEEE M = {M}, x={xsol}")
    print(f"strain {np.diff(xsol)}")
    plt.plot((np.array(hist)-correctFuncMin)/np.abs(correctFuncMin),label=f"convergence history M = {M}")

plt.xlabel("Iterations")
plt.ylabel("Function value")
plt.yscale("log")
plt.legend()
plt.savefig(f"N-{N}-linear-case.svg",bbox_inches="tight")
plt.legend()


plt.figure(2)

N = 100
funcObj = ue.IncompressibleMooneyRivlinFunc(N, Umax=2.,E=np.ones(N+1)*1e3)
correctFuncMin = funcObj.trueMinFunc()

x0 = funcObj.getInitialGuess()
Niter = int(1.5*N)
# gradient descent
xsol, hist, alphas = ma.gradientDescent(x0, Niter, funcObj)
plt.plot((np.array(hist)-correctFuncMin)/np.abs(correctFuncMin),'.--',label="Gradient descent")
for updateMethod in ["Hestenes and Stiefel 1952", "Fletcher and Reeves 1964", 
                      "Polak and Ribiere 1969", "Fletcher 1987",
                      "Liu and Storey 1991", "Dai and Yuan 1999", "Hager and Zhang 2005"]:
    print(f"running updateMethod = {updateMethod}")                     
    xsol, hist, alphas = ma.conjugateGradient(x0, Niter, funcObj, updateMethod)
    print(f"updateMethod = {updateMethod}, x={xsol}")
    plt.plot((np.array(hist)-correctFuncMin)/np.abs(correctFuncMin),'.--',label=f"CG - {updateMethod}")

Mlist = [1,2,10,50, N]
for M in Mlist:
    print(f"RUNNING  M = {M}")
    xsol, hist, alphas = ma.multipleDirectionSearch(x0, Niter, funcObj, M=M)
    print(f"NEEEEEEEE M = {M}, x={xsol}")
    print(f"strain {np.diff(xsol)}")
    plt.plot((np.array(hist)-correctFuncMin)/np.abs(correctFuncMin),label=f"convergence history M = {M}")
    
plt.xlabel("Iterations")
plt.ylabel("Function value")
plt.yscale("log")
plt.legend()
plt.savefig(f"N-{N}-nonlinear-case.svg",bbox_inches="tight")
plt.legend()


plt.show()
