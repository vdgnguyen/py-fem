
import numpy as np

from dwave.system import LeapHybridSampler
from dimod import BinaryQuadraticModel


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
        
class DWaveQUBO(QUBO):
    def __init__(self, optType):
        self.optType=optType
        
    def solve(self, h, J):
        print("start optimising ...")
        numBits = len(h)
        x = ["v"+str(p) for p in range(numBits)]
        linear = {}
        quadratic = {}
        offset = 0.
        bqm = BinaryQuadraticModel(vartype=self.optType)
        if self.optType=="SPIN":
            for i in range(numBits):
                offset += 0.5*J[i,i]
                linear[x[i]] = h[i]
                for j in range(i+1,numBits):
                    quadratic[(x[i],x[j])]=0.5*(J[i,j]+J[j,i])      
        elif self.optType =="BINARY":
            for i in range(numBits):
                linear[x[i]] = h[i] +0.5*J[i,i] 
                for j in range(i+1,numBits):
                    quadratic[(x[i],x[j])]=0.5*(J[i,j]+J[j,i])
        else:
            raise NotImplementedError(f"{self.optType} has not been implemented")
        
        for i in range(numBits):
            bqm.add_linear(i,  linear[x[i]])
            for j in range(i+1, numBits):
                bqm.add_quadratic(i, j, quadratic[(x[i],x[j])])
        #print("BQM: ", bqm)
        
        sampler = LeapHybridSampler()
        sampleset = sampler.sample(bqm)
        print("sampleset=",sampleset)
        sample = sampleset.first.sample
        print("sample",sample)
        print("done optimising !!!")
        qOpt = [sample[i] for i in range(numBits)]
        return qOpt, sampleset.first.energy+offset
        
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
        elif self.optType =="BINARY":
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
        else:
            raise NotImplementedError(f"{self.optType} has not been implemented")
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
    
    qubo2 = DWaveQUBO("SPIN")
    qOpt, fOpt = qubo2.solve(h,J)
    print(f"Solution: {qOpt}: func {fOpt}")

    print("*********************************************")

    qubo = NaiveQUBO("BINARY")
    qOpt, fOpt = qubo.solve(h,J)
    print(f"Solution: {qOpt}: func {fOpt}")
    
    qubo2 = DWaveQUBO("BINARY")
    qOpt, fOpt = qubo2.solve(h,J)
    print(f"Solution: {qOpt}: func {fOpt}")

