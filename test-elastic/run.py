import sys
sys.path.insert(0, '../src')
import pandas as pd
import matplotlib.pyplot as plt
import Solver

mysolver = Solver.FEMSolver("ElasticTest")
mysolver.loadMeshFile("input-elastic.dat")
mysolver.solve(nsteps=1, maxNRIterations=100,lineSearch=False)

a = pd.read_csv("ElasticTest-lowerBound-comp1.csv")
plt.figure()
plt.plot(a.values[:,0],-a.values[:,1],'r.-')
plt.xlabel("Time (s)")
plt.ylabel("Force (N)")
plt.show()
