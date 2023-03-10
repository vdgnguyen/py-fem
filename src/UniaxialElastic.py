import numpy as np

class QuadraticFunc:
    def __init__(self, A, b):
        self.A = A
        self.b = b
    
    def getInitialGuess(self):
        return self.b/np.diag(self.A)
        
    def func(self,x):
        return 0.5*np.dot(x,np.matmul(self.A,x)) - np.dot(self.b,x)

    def gradFunc(self,x):
        return np.matmul(self.A,x) - self.b
        
    def hessFunc(self,x):
        return self.A
        
    def trueMinFunc(self):
        xsol = np.linalg.solve(self.A, self.b)
        return self.func(xsol)
        
class LinearElasticFunc:
    def __init__(self, N, Umax=2., E=None):
        self.N =N
        self.Umax=Umax
        self.L=1./(N+1)
        if E is None:
            self.E =np.ones(N+1)*1e3
        else:
            self.E  = E
    def getInitialGuess(self):
        x0 = np.zeros(self.N)
        b = self.gradFunc(x0)
        H = self.hessFunc(x0)
        return -b/np.diag(H)
        
    def func(self,x):
        allX = [0]+x.tolist()+[self.Umax]
        func = 0
        for i in range(self.N+1):
            lam = 1+(allX[i+1]-allX[i])/self.L
            func += self.L*0.5*self.E[i]*(lam-1)*(lam-1)
        return func

    def gradFunc(self,x):
        allX = [0]+x.tolist()+[self.Umax]
        grad = np.zeros(self.N)
        for i in range(self.N+1):
            lam = 1+(allX[i+1]-allX[i])/self.L
            if i==0:
                grad[0] += self.L*self.E[i]*(lam-1)*(1./self.L)
            elif i == self.N:
                grad[self.N-1] += self.L*self.E[i]*(lam-1)*(-1/self.L)
            else:
                grad[i-1] += self.L*self.E[i]*(lam-1)*(-1/self.L)
                grad[i] += self.L*self.E[i]*(lam-1)*(1/self.L)
        return grad
        
    def hessFunc(self,x):
        allX = [0]+x.tolist()+[self.Umax]
        H = np.zeros((self.N,self.N))
        for i in range(self.N+1):
            lam = 1+(allX[i+1]-allX[i])/self.L
            if i==0:
                H[0,0] += self.L*self.E[i]*(1)*(1/self.L)*(1/self.L)
            elif i == self.N:
                H[self.N-1,self.N-1] += self.L*self.E[i]*(1)*(1/self.L)*(1/self.L)
            else:
                H[i-1,i-1] += self.L*self.E[i]*(1)*(-1/self.L)*(-1/self.L)
                H[i-1,i] +=  self.L*self.E[i]*(1)*(-1/self.L)*(1/self.L)
                H[i,i-1] += self.L*self.E[i]*(1)*(1/self.L)*(-1/self.L)
                H[i,i] += self.L*self.E[i]*(1)*(1/self.L)*(1/self.L)
        return H
        
    def trueMinFunc(self):
        x = np.zeros(self.N)
        tol = 1e-12
        abstol = 1e-12
        norm0 = None
        iterIndex=0
        while True:
            iterIndex +=1
            if iterIndex > 50:
                raise RuntimeError("solution does not converge")
            g = self.gradFunc(x)
            norm = np.linalg.norm(g,np.inf)    
            if norm0 is None:
                norm0 = norm
            print(f"ITER {iterIndex}: relative error = {norm/norm0:.6e} absolute error = {norm:.6e}")
            if norm < abstol:
                print("convergence by absolute tolerance")
                return self.func(x)
            elif norm< tol* norm0:
                print("convergence by relative tolerance")
                print("unknown=",x)
                print("strain=",np.diff(np.array([0]+x.tolist()+[self.Umax]))/self.L)
                print("stress=",self.E*np.diff(np.array([0]+x.tolist()+[self.Umax]))/self.L)
                return self.func(x)
            else:
                # update H
                H = self.hessFunc(x)
                x -= np.linalg.solve(H, g)
        
        
class IncompressibleMooneyRivlinFunc:
    def __init__(self, N, Umax=2., E=None):
        self.N =N
        self.Umax=Umax
        self.L=1./(N+1)
        if E is None:
            self.E =np.ones(N+1)*1e3
        else:
            self.E  = E
        
    def getInitialGuess(self):
        x0 = np.zeros(self.N)
        b = self.gradFunc(x0)
        H = self.hessFunc(x0)
        return -b/np.diag(H)
        
    def func(self,x):
        allX = [0]+x.tolist()+[self.Umax]
        func = 0
        for i in range(self.N+1):
            lam = 1+(allX[i+1]-allX[i])/self.L
            func += self.L*self.E[i]*(0.5*lam**2+1./lam-3./2.)
        return func

    def gradFunc(self,x):
        allX = [0]+x.tolist()+[self.Umax]
        grad = np.zeros(self.N)
        for i in range(self.N+1):
            lam = 1+(allX[i+1]-allX[i])/self.L
            if i==0:
                grad[0] += self.L*self.E[i]*(lam-1./lam/lam)*(1./self.L)
            elif i == self.N:
                grad[self.N-1] += self.L*self.E[i]*(lam-1./lam/lam)*(-1/self.L)
            else:
                grad[i-1] += self.L*self.E[i]*(lam-1./lam/lam)*(-1/self.L)
                grad[i] += self.L*self.E[i]*(lam-1./lam/lam)*(1/self.L)
        return grad
        
    def hessFunc(self,x):
        allX = [0]+x.tolist()+[self.Umax]
        H = np.zeros((self.N,self.N))
        for i in range(self.N+1):
            lam = 1+(allX[i+1]-allX[i])/self.L
            if i==0:
                H[0,0] += self.L*self.E[i]*(1+2./lam/lam/lam)*(1/self.L)*(1/self.L)
            elif i == self.N:
                H[self.N-1,self.N-1] += self.L*self.E[i]*(1+2./lam/lam/lam)*(1/self.L)*(1/self.L)
            else:
                H[i-1,i-1] += self.L*self.E[i]*(1+2./lam/lam/lam)*(-1/self.L)*(-1/self.L)
                H[i-1,i] +=  self.L*self.E[i]*(1+2./lam/lam/lam)*(-1/self.L)*(1/self.L)
                H[i,i-1] += self.L*self.E[i]*(1+2./lam/lam/lam)*(1/self.L)*(-1/self.L)
                H[i,i] += self.L*self.E[i]*(1+2./lam/lam/lam)*(1/self.L)*(1/self.L)
        return H
        
    def trueMinFunc(self):
        x = np.zeros(self.N)
        tol = 1e-12
        abstol = 1e-12
        norm0 = None
        iterIndex=0
        while True:
            iterIndex +=1
            if iterIndex > 50:
                raise RuntimeError("solution does not converge")
            g = self.gradFunc(x)
            norm = np.linalg.norm(g,np.inf)    
            if norm0 is None:
                norm0 = norm
            print(f"ITER {iterIndex}: relative error = {norm/norm0:.6e} absolute error = {norm:.6e}")
            if norm < abstol:
                print("convergence by absolute tolerance")
                return self.func(x)
            elif norm< tol* norm0:
                print("convergence by relative tolerance")
                print("unknown=",x)
                alll=1+np.diff(np.array([0]+x.tolist()+[self.Umax]))/self.L
                sig = self.E*(alll-1/alll**2)
                print("strain=",alll)
                print("stress=",sig)
                return self.func(x)
            else:
                # update H
                H = self.hessFunc(x)
                x -= np.linalg.solve(H, g)
        
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    vv = np.linspace(0,3, 10)
    minElastic = vv*0
    minNonLinear = vv*0
    N = 100
    E = (np.random.rand(N+1)+1)*1e3
    for i in range(1,len(vv)):
        func = LinearElasticFunc(N, Umax=vv[i],E=E)
        minFunc = func.trueMinFunc()
        print(f"minimum value = {minFunc}")
        minElastic[i]=minFunc
        
        
        func = IncompressibleMooneyRivlinFunc(N, Umax=vv[i], E=E)
        minFunc = func.trueMinFunc()
        print(f"minimum value = {minFunc}")
        minNonLinear[i]=minFunc
        
    
    plt.figure()
    plt.plot(vv,minElastic,'r.-',label="linear elastic")
    plt.plot(vv,minNonLinear,'g.-',label="linear elastic")
    plt.xlabel("U prescribed")
    plt.ylabel("minimum energy")
    plt.show()
