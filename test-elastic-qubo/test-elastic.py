
import sys
sys.path.insert(0, '../src')
import pandas as pd
import matplotlib.pyplot as plt
import Solver
import QUBO

#qubo=QUBO.NaiveQUBO("SPIN")

qubo = QUBO.DWaveQUBO("SPIN")
mysolver = Solver.FEMSolver("ElasticTest")
mysolver.loadMeshFile("input-elastic.dat")
print(f"number of dofs = {len(mysolver.dofDict)}")
mysolver.qubo_solve(qubo, nsteps=1, maxNRIterations=5000,
                    nBitsGradient=0,
                    etaMin = 0.,
                    etaMax = 1e-1,
                    nBitsRandom =15,
                    alpha = 1e-3,
                    tol = 1e-3,
                    maxNumIncreased=5,
                    increasedFactor=1.25,
                    maxNumReduced=2,
                    reducedFactor=0.75,)

a = pd.read_csv("ElasticTest-lowerBound-comp1.csv")
plt.figure()
plt.plot(a.values[:,0],-a.values[:,1],'r.-')
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.show()

