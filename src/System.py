"""
Linear and nonlinear system to solve the FE problem
"""

import numpy as np
from scipy.sparse import csc_matrix, linalg
import time
import QUBO


class linearSystem:
    def __init__(self, systemSize):
        """
        initialise the linear system
        Args:
          systemSize (int): size of the linear system
        Returns:
          None
        """
        self.systemSize=systemSize
        self.A = {}
        self.b = np.zeros(systemSize)
        self.x = np.zeros(systemSize)
    def updateParameters(self, params):
        pass
    def withStiffnessMatrix(self):
        return True
    
    def zeroMatrix(self):
        for kk in self.A.keys():
            self.A[kk] = 0.
      
    def addToMatrix(self, i, j, value):
        """
        Add an element to the global stiffness matrix
        Args:
          i (int): row index
          j (int): col index
          val(float): value
        Returns:
          None
        """
        if (i,j) in self.A.keys():
            self.A[(i,j)] += value
        else:
            self.A[(i,j)] = value
      
    def zeroRHS(self):
        """
        Zero RHS vector
        Args:
          None
        Returns:
          None
        """
        self.b.fill(0.)
        
    def addToRHS(self, i, value):
        """
        Add an element to the global rhs vector
        Args:
          i (int): row index
          val(float): value
        Returns:
          None
        """
        self.b[i]+= value
    
    def normInfRHS(self):
        return np.linalg.norm(self.b)
    
    def printSystem(self):
        print("A",self.A)
        print("rhs",self.b)
        print("x",self.x)
        
    def systemSolve(self):
        """
        Solve linear system
        Args:
          None
        Returns:
          None
        """
        start_time = time.time()
        numEle=len(self.A)
        row = np.zeros(numEle,dtype=int)
        col = np.zeros(numEle,dtype=int)
        data = np.zeros(numEle,dtype=float)
        ite=0
        for pos, val in self.A.items():
          row[ite] = pos[0]
          col[ite] = pos[1]
          data[ite] = val
          ite = ite+1
         
        Amat = csc_matrix((data, (row, col)), shape=(self.systemSize, self.systemSize))
        lu = linalg.splu(Amat)
        self.x = lu.solve(self.b)    
        print("solving time in linearSystem: %e seconds" % (time.time() - start_time))
        return 1
    
    def updateSolution(self, acceptFlag):
        pass

    def getSolution(self, i):
        """
        get an element from the solution
        Args:
          i (int): row index
        Returns:
          solution (float)
        """
        return self.x[i]
        
class nonLinearSystem(linearSystem):
    def __init__(self, systemSize, fullSize):
        """
        initialise the linear system
        Args:
          systemSize (int): size of the linear system
          fullSize (int): size of the total dof without accounting the dirichlet boundary conditions
        Returns:
          None
        """
        super().__init__(systemSize)
        self.xcur = np.zeros(systemSize)
        self.xprev = np.zeros(systemSize)
        self.Fint =  np.zeros(systemSize)
        self.Fext =  np.zeros(systemSize)
        
        self.fullSize=fullSize
        self.FintFull =  np.zeros(fullSize)
        self.FextFull =  np.zeros(fullSize)
    
    def withStiffnessMatrix(self):
        return True
    
    def zeroRHS(self):
        """
        Zero RHS vector
        Args:
          None
        Returns:
          None
        """
        super().zeroRHS()
        self.Fint.fill(0)
        self.Fext.fill(0)
        self.FintFull.fill(0)
        self.FextFull.fill(0)
    
    def addToFintFull(self, i, val):
        """
        Add an element to the global internal force vector
        Args:
          i (int): row index
          val(float): value
        Returns:
          None
        """
        self.FintFull[i] += val
        
    
    def addToFextFull(self, i, val):
        """
        Add an element to the global external force vector
        Args:
          i (int): row index
          val(float): value
        Returns:
          None
        """
        self.FextFull[i] += val
        
    def addToFint(self, i, val):
        """
        Add an element to the global internal force vector
        Args:
          i (int): row index
          val(float): value
        Returns:
          None
        """
        self.Fint[i] += val
        
    
    def addToFext(self, i, val):
        """
        Add an element to the global external force vector
        Args:
          i (int): row index
          val(float): value
        Returns:
          None
        """
        self.Fext[i] += val
        
    def normInfRHS(self):
        return np.linalg.norm(self.Fint - self.Fext)

    def normInfFint(self):
        return np.linalg.norm(self.Fint)
  
    def normInfFext(self):
        return np.linalg.norm(self.Fext)
                            
    
    def printSystem(self):
        super().printSystem()
        print("Fint",self.Fint)
        print("Fext",self.Fext)
        print("xcur",self.xcur)
        print("xprev",self.xprev)
        
    def systemSolve(self):
        """
        Solve linear system
        Args:
          None
        Returns:
          None
        """
        start_time = time.time()
        self.b = self.Fint-self.Fext
        ok = super().systemSolve()
        print("solving time in nonLinearSystem: %e seconds" % (time.time() - start_time))
        return ok
    
    def updateSolution(self, acceptFlag):
        if acceptFlag:
            self.xcur -= self.x
    
    def nextStep(self):
        """
        store data when making next time step
        """
        self.xprev[:] =self.xcur
      
    def resetToPrevousStep(self):
        """
        reset data to previous time step
        """
        self.xcur[:] = self.xprev
        
    def getSolution(self, i):
        """
        get an element from the solution
        Args:
          i (int): row index
        Returns:
          solution (float)
        """
        return self.xcur[i] 

class lineSearchSystem(nonLinearSystem):
    def __init__(self, systemSize, fullSize, beta=0.):
        super().__init__(systemSize, fullSize)
        self.alpha=0.
        self.xtemp=np.zeros(systemSize)
        self.beta=beta
    
    def updateSolution(self, acceptFlag):
        if acceptFlag:
            self.xtemp[:] =self.xcur
            self.xcur -= self.x
            
    def systemSolve(self):
        self.b = self.Fint-self.Fext
        numEle=len(self.A)
        row = np.zeros(numEle,dtype=int)
        col = np.zeros(numEle,dtype=int)
        data = np.zeros(numEle,dtype=float)
        ite=0
        for pos, val in self.A.items():
          row[ite] = pos[0]
          col[ite] = pos[1]
          data[ite] = val
          ite = ite+1
        Amat = csc_matrix((data, (row, col)), shape=(self.systemSize, self.systemSize))
        self.alpha = np.dot(self.b,self.b)/ np.dot(self.b,Amat.dot(self.b))
        print(f"alpha = {self.alpha} beta={self.beta}")
        
        self.x = (self.alpha*self.b - self.beta*(self.xcur-self.xtemp))
        
        return 1

class nonlinearConjugateGradientSystem(nonLinearSystem):
    def __init__(self, systemSize, fullSize):
        super().__init__(systemSize, fullSize)
        self.alpha=0.
        self.xtemp=np.zeros(systemSize)
        self.bprev = np.zeros(systemSize)
        self.d = np.zeros(systemSize)
        self.dprev = np.zeros(systemSize)
        self.first=False
        
    def zeroRHS(self):
        self.bprev[:] = self.b
        self.dprev[:] = self.d
        super().zeroRHS()
    
    def updateSolution(self, acceptFlag):
        self.first=True
        if acceptFlag:
            self.xtemp[:] =self.xcur
            self.xcur -= self.x
    
    def nextStep(self):
        """
        store data when making next time step
        """
        super().nextStep()
        self.first=False
    
    def resetToPrevousStep(self):
        super().resetToPrevousStep()
        self.first=False

    def systemSolve(self):
        self.b = self.Fint-self.Fext
        beta=1.
        if self.first:
            beta = np.dot(self.b,self.b)/np.dot(self.bprev,self.bprev)
        self.d = -1*self.b + beta*self.dprev
        numEle=len(self.A)
        row = np.zeros(numEle,dtype=int)
        col = np.zeros(numEle,dtype=int)
        data = np.zeros(numEle,dtype=float)
        ite=0
        for pos, val in self.A.items():
          row[ite] = pos[0]
          col[ite] = pos[1]
          data[ite] = val
          ite = ite+1
        Amat = csc_matrix((data, (row, col)), shape=(self.systemSize, self.systemSize))
        self.alpha = np.dot(self.b,self.d)/ np.dot(self.d,Amat.dot(self.d))
        print(f"alpha = {self.alpha} beta={beta}")
        self.x = (self.alpha*self.d)
        return 1

    
class binarySearchSystem(nonLinearSystem):
    def __init__(self, solver, qubo, systemSize, fullSize, 
                 nBitsGradient, etaMin, etaMax, 
                 nBitsRandom, alpha):
        super().__init__(systemSize, fullSize)
        self.solver = solver
        self.qubo = qubo
        self.nBitsGradient = nBitsGradient
        self.etaMin = etaMin
        self.etaMax = etaMax
        self.nBitsRandom=nBitsRandom
        self.alpha=alpha
        self.directionTmp = np.zeros(systemSize)
        # temp solution
        self.xtemp = np.zeros(systemSize)
        self.initialGradientNorm = None
        if self.nBitsGradient >0:
            self.beta=np.zeros(self.nBitsGradient)
            fact=1./(2**self.nBitsGradient-1.)
            for i in range(self.nBitsGradient):
                self.beta[i]=-(self.etaMax-self.etaMin)*(2**(i-1))*fact
            print("betta=",self.beta)
        else:
            self.beta=None
    
    def withStiffnessMatrix(self):
        return False
    
    def updateParameters(self, fact):
        self.etaMin *= fact
        self.etaMax *=fact
        self.alpha *=fact
        if fact < 1:
            print(f"REDUCING FACTORS from {self.etaMin/fact} {self.etaMax/fact} {self.alpha/fact} to {self.etaMin} {self.etaMax} {self.alpha}")
        else:
            print(f"INCREASING FACTORS from {self.etaMin/fact} {self.etaMax/fact} {self.alpha/fact} to {self.etaMin} {self.etaMax} {self.alpha}")
        
        if self.nBitsGradient >0:
            self.beta=np.zeros(self.nBitsGradient)
            fact=1./(2**self.nBitsGradient-1.)
            for i in range(self.nBitsGradient):
                self.beta[i]=-(self.etaMax-self.etaMin)*(2**(i-1))*fact
            print("beta modified=",self.beta)
            
    def updateSolution(self, acceptFlag):
        if not(acceptFlag):
            self.xcur[:] = self.xtemp
        else:
            self.directionTmp = self.b

    def systemSolve(self):
        # save current solution
        self.xtemp[:]=self.xcur
        # define a and D
        # direction search
        if self.initialGradientNorm is None:
            self.initialGradientNorm=np.linalg.norm(self.Fint - self.Fext)
        if self.nBitsRandom >0:
            D=np.zeros((self.systemSize,self.nBitsRandom))
            for i in range(self.nBitsRandom):
                v =  2*np.random.rand(self.systemSize)-1
                vnorm = np.linalg.norm(v)
                nRHS = np.linalg.norm(self.Fint - self.Fext)
                D[:,i]= self.alpha*(v/vnorm) # + (self.Fint-self.Fext)/nRHS)
            # create and solve QUBO problem
            h, J = QUBO.QUBO.createQUBO(self.solver,a=None,D=D)
            bSol = self.qubo.solve(h,J)
            print(bSol)
            self.b = -np.dot(D,bSol[0])
            norm = np.linalg.norm(self.b,np.inf)
            self.b *= (1/norm)
        else:
            self.b = (self.Fint-self.Fext)/self.initialGradientNorm
            
            fact =np.dot(self.b,self.b)/(np.dot(self.directionTmp,self.directionTmp))
            g = -self.b + fact*self.directionTmp
            
        # line search
        a = -0.5*(self.etaMin+self.etaMax)*self.b
        D = np.zeros((self.systemSize, self.nBitsGradient))
        for i in range(self.nBitsGradient):
            D[:,i] = self.beta[i]*self.b
            
        h, J = QUBO.QUBO.createQUBO(self.solver,a,D)
        bSol = self.qubo.solve(h,J)
        print(bSol)
        eta = 0.5*(self.etaMin+self.etaMax) - np.dot(self.beta,bSol[0][:self.nBitsGradient])
        print("eta=", eta)
        if eta < 0:
            raise RuntimeError("Eta cannot be negative")
        self.xcur += (a + np.dot(D,bSol[0]))
        return 1
    
class gradientDescentSystem(nonLinearSystem):
    def __init__(self, solver, qubo, systemSize, fullSize, 
                 nBitsGradient, etaMin, etaMax, 
                 nBitsRandom, alpha):
        super().__init__(systemSize, fullSize)
        self.solver = solver
        self.qubo = qubo
        self.nBitsGradient = nBitsGradient
        self.etaMin = etaMin
        self.etaMax = etaMax
        self.nBitsRandom=nBitsRandom
        self.alpha=alpha
        # temp solution
        self.xtemp = np.zeros(systemSize)
        self.initialGradientNorm = None
        if self.nBitsGradient >0:
            self.beta=np.zeros(self.nBitsGradient)
            fact=1./(2**self.nBitsGradient-1.)
            for i in range(self.nBitsGradient):
                self.beta[i]=-(self.etaMax-self.etaMin)*(2**(i-1))*fact
            print("betta=",self.beta)
        else:
            self.beta=None
    
    def withStiffnessMatrix(self):
        return False
    
    def updateParameters(self, fact):
        self.etaMin *= fact
        self.etaMax *=fact
        self.alpha *=fact
        if fact < 1:
            print(f"REDUCING FACTORS from {self.etaMin/fact} {self.etaMax/fact} {self.alpha/fact} to {self.etaMin} {self.etaMax} {self.alpha}")
        else:
            print(f"INCREASING FACTORS from {self.etaMin/fact} {self.etaMax/fact} {self.alpha/fact} to {self.etaMin} {self.etaMax} {self.alpha}")
        
        if self.nBitsGradient >0:
            self.beta=np.zeros(self.nBitsGradient)
            fact=1./(2**self.nBitsGradient-1.)
            for i in range(self.nBitsGradient):
                self.beta[i]=-(self.etaMax-self.etaMin)*(2**(i-1))*fact
            print("beta modified=",self.beta)
            
    def updateSolution(self, acceptFlag):
        if not(acceptFlag):
            self.xcur[:] = self.xtemp

    def systemSolve(self):
        # save current solution
        self.xtemp[:]=self.xcur
        # define a and D
        #b = fint -fext is the gradient of energy
        # gradient of the total energy
        if self.nBitsGradient == 0 and self.nBitsRandom >0:
            D=np.zeros((self.systemSize,self.nBitsRandom))
            for i in range(self.nBitsRandom):
                v =  2*np.random.rand(self.systemSize)-1
                vnorm = np.linalg.norm(v)
                D[:,i]= self.alpha*(v/vnorm)
            # create and solve QUBO problem
            h, J = QUBO.QUBO.createQUBO(self.solver,a=None,D=D)
            bSol = self.qubo.solve(h,J)
            print("PASS HERE", bSol)
            self.xcur += np.dot(D,bSol[0])
        elif self.nBitsGradient > 0:
            self.b = self.Fint-self.Fext 
            if self.initialGradientNorm is None:
                self.initialGradientNorm= np.linalg.norm(self.b,np.inf)
            self.b *= (1./self.initialGradientNorm)
            a = -0.5*(self.etaMin+self.etaMax)*self.b
            D = np.zeros((self.systemSize, self.nBitsGradient+self.nBitsRandom))
            for i in range(self.nBitsGradient):
                D[:,i] = self.beta[i]*self.b
            if self.nBitsRandom >0:
                D[:,self.nBitsGradient:]=self.alpha*(2*np.random.rand(self.systemSize,self.nBitsRandom)-1)
            # create and solve QUBO problem
            h, J = QUBO.QUBO.createQUBO(self.solver,a,D)
            bSol = self.qubo.solve(h,J)
            print(bSol)
            if self.nBitsGradient  > 0:
                eta = 0.5*(self.etaMin+self.etaMax) - np.dot(self.beta,bSol[0][:self.nBitsGradient])
                print("eta=", eta)
                if eta < 0:
                    raise RuntimeError("Eta cannot be negative")
            self.xcur += (a + np.dot(D,bSol[0]))
        else:
            raise NotImplementedError("This case is not implemented")
        return 1
    
if __name__ == "__main__":
    print("testing")
    import matplotlib.pyplot as plt
    
    np.random.seed(42)
    plt.close("all")
    plt.figure()

    N = 300
    A = np.zeros((N,N))
    b = np.zeros(N)
    eig= np.array([i+1 for i in range(N)])
    for i in range(N):
        b[i]=np.random.rand()
        v = np.random.rand(N)
        v *= (1./np.linalg.norm(v))
        A += eig[i]*np.outer(v,v)
    
    
    xsol = np.linalg.solve(A, b)
    correct=  0.5*np.dot(xsol,np.dot(A,xsol)) - np.dot(b,xsol)+1000
    print(f"correct solution = {correct}")
    for beta in [0.5]:
        #lsys = lineSearchSystem(N, N+1, beta)
        #lsys = nonLinearSystem(N,N+1)
        lsys = nonlinearConjugateGradientSystem(N,N+1)
        Nsteps =5*N
        res=[]
        for it in range(Nsteps): 
            x = (lsys.xcur)
            func = 0.5*np.dot(x,np.dot(A,x)) - np.dot(b,x)+1000
            grad = np.dot(A,x)-b
            normGrad=np.linalg.norm(grad,np.inf)
            res.append([func, normGrad])
            print(f"ITR {it}: func = {func} gradNorm = {normGrad}")
            if normGrad < 1e-10:
                break
            lsys.zeroMatrix()
            for i in range(N):
                for j in range(N):
                    if A[i,j] !=0.:
                        lsys.addToMatrix(i,j, A[i,j])
            lsys.zeroRHS()
            for i in range(N):
                lsys.addToFint(i, grad[i])
            lsys.systemSolve()
            lsys.updateSolution(True)
        res=np.array(res)
        plt.plot((res[:,0]-correct),".-", label=f"beta={beta}")
    #plt.yscale("log")
    plt.legend()
    plt.show()
