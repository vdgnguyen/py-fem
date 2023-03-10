import sys
sys.path.insert(0, '../src')
import numpy as np
import matplotlib.pyplot as plt
import UniaxialElastic as ue
import MinimisationAlgorithm as ma

Amat = np.genfromtxt('A.csv', delimiter=',')
bvec = np.genfromtxt('b.csv', delimiter=',')
N = len(bvec)
funcObj = ue.QuadraticFunc(Amat,bvec)
correctFuncMin=funcObj.trueMinFunc()
x0 = funcObj.getInitialGuess()

plt.close("all")
plt.figure(1)


x0 = funcObj.getInitialGuess()
Niter = int(1.5*N)
# gradient descent
xsol, hist, alphas =  ma.gradientDescent(x0, Niter, funcObj)
plt.plot((np.array(hist)-correctFuncMin),'.--',label="Gradient descent")
for updateMethod in ["Hestenes and Stiefel 1952", "Fletcher and Reeves 1964", 
                      "Polak and Ribiere 1969", "Fletcher 1987",
                      "Liu and Storey 1991", "Dai and Yuan 1999", "Hager and Zhang 2005"]:
    print(f"running updateMethod = {updateMethod}")                     
    xsol, hist, alphas =  ma.conjugateGradient(x0, Niter, funcObj, updateMethod)
    print(f"updateMethod = {updateMethod}, x={xsol}")
    plt.plot((np.array(hist)-correctFuncMin),'.--',label=f"CG - {updateMethod}")

Mlist = [1,2,10,50, N]
for M in Mlist:
    print(f"RUNNING  M = {M}")
    xsol, hist, alphas = ma.multipleDirectionSearch(x0, Niter, funcObj, M=M)
    print(f"NEEEEEEEE M = {M}, x={xsol}")
    print(f"strain {np.diff(xsol)}")
    plt.plot((np.array(hist)-correctFuncMin),label=f"convergence history M = {M}")

plt.xlabel("Iterations")
plt.ylabel("Function value")
plt.yscale("log")
plt.savefig("comparision.svg", bbox_inches="tight")
plt.legend()
