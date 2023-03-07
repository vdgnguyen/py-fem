
import numpy as np


class QUBO :
    def __init__(self):
        pass
    
    def solve(self, h, J):
        pass
    
    @staticmethod
    def createQUBO(solver, a=None, D=None):
        nBits = D.shape[1]
        h = np.zeros(nBits)
        J = np.zeros((nBits,nBits))
        for GP in solver.GPs:
            # local gradient and hessian
            ge = GP.currentState.internalForce - GP.currentState.externalForce 
            Ke = GP.currentState.stiffness
            #
            localPos=[]
            globalPos=[]
            for i in range(GP.totalDofs):
                if GP.localDofs[i] not in solver.fixedDofs:
                    localPos.append(i)
                    globalPos.append(solver.dofDict[GP.localDofs[i]])
            if len(localPos) > 0:
                ae = np.zeros(GP.totalDofs)
                De = np.zeros((GP.totalDofs,nBits))
                if a is not None:
                    ae[localPos] = a[globalPos]
                
                if D is not None:
                    De[localPos,:] =D[globalPos]
              
                h += np.dot(ge.T + np.dot(ae.T,Ke),De)
                J += np.dot(np.dot(De.T,Ke),De)
        return h, J
        
class NaiveQUBO(QUBO):
    def __init__(self, optType):
        """

        Parameters
        ----------
        optType : String
            beeteen SPIN (opt variable in -1 and 1) or BIN (opt variable in 0 and 1).

        Returns
        -------
        None.

        """
        self.optType=optType

    def solve(self, h, J):
        print("start optimising ...")
        numBits = len(h)
        numCases=2**numBits
        if self.optType=="SPIN":
            qOpt = 2*np.zeros(numBits)-1.
            fOpt = np.dot(qOpt,h) + 0.5*np.dot(qOpt,np.dot(J,qOpt))
            for ic in range(1,numCases):
                bitString = '{:b}'.format(ic)
                while len(bitString) < numBits:
                    bitString ="0"+bitString
                q = np.array([2*float(a)-1 for a in bitString])
                f = np.dot(q,h) + 0.5*np.dot(q,np.dot(J,q))
                #print(f"{q}: func {f}")
                if f < fOpt:
                    fOpt=f
                    qOpt[:]= q
        elif self.optType =="BIN":
            qOpt = np.zeros(numBits)
            fOpt = np.dot(qOpt,h) + 0.5*np.dot(qOpt,np.dot(J,qOpt))
            for ic in range(1,numCases):
                bitString = '{:b}'.format(ic)
                while len(bitString) < numBits:
                    bitString ="0"+bitString
                q = np.array([float(a) for a in bitString])
                f = np.dot(q,h) + 0.5*np.dot(q,np.dot(J,q))
                #print(f"{q}: func {f}")
                if f < fOpt:
                    fOpt=f
                    qOpt[:]= q
        print("done optimising !!!")
        return qOpt, fOpt
        

if __name__ == "__main__":
    
    #np.random.seed(42)
    nBits = 5
    h = np.random.rand(nBits)
    J = np.random.rand(nBits,nBits)*2-1
    J = 0.5*(J + np.transpose(J))
    print(h)
    print(J)
    
    qubo = NaiveQUBO("SPIN")
    qOpt, fOpt = qubo.solve(h,J)
    print(f"Solution: {qOpt}: func {fOpt}")
    
    qubo = NaiveQUBO("BIN")
    qOpt, fOpt = qubo.solve(h,J)
    print(f"Solution: {qOpt}: func {fOpt}")
