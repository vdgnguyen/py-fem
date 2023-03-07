import numpy as np

class MaterialLaw :
    def __init__(self):
        self.locVec2D= [3*i+j for i in range(2) for j in range(2)]
        self.locMat2D = np.ix_(self.locVec2D,self.locVec2D)
    
    def allocateInternalState(self, GP):
        """
        allocate internal state
        Parameters
        ----------
        GP : GaussPoint
            data at GP.

        Returns
        -------
        None.

        """
        pass
        
    def constitutive(self, GP, dispVec, curTime, timeStep, stiff):
        """
        evaluate the constitutive relation at GP

        Parameters
        ----------
        GP : GaussPoint
            all data in a GP.
        dispVec : numpy array
            displacement at this GP.
        curTime : float
            current time.
        timeStep : float
            time step.
        stiff : boolean
            True if tangent needs to be computed.

        Returns
        -------
        None.

        """
        pass
    
    def computeInternalForce(self, GP):
        """
        compute internal force at Gausspoint

        Parameters
        ----------
        GP : GaussPoint

        Returns
        -------
        None.

        """
        B = GP.shapeFuncGrad
        GP.currentState.internalForce = GP.wJ*np.matmul(B.T,GP.currentState.fluxFields)
        
    def computeStiffness(self, GP):
        """
        compute stiffness at Gausspoint

        Parameters
        ----------
        GP : GaussPoint

        Returns
        -------
        None.

        """
        B = GP.shapeFuncGrad
        GP.currentState.stiffness = GP.wJ*np.matmul(B.T,np.matmul(GP.currentState.tangent,B))
        
class LinearElasticLaw(MaterialLaw):
    def __init__(self, lawName, E, nu):
        """
        linear elastic material law

        Parameters
        ----------
        lawName : string
            material name.
        E : float
            Young modulus.
        nu : float
            Poisson ratio.

        Returns
        -------
        None.

        """
        super().__init__()
        self.lawName=lawName
        self.E = E 
        self.nu = nu
        self.lambd = E*nu/(1+nu)/(1-2*nu)
        self.mu = E/2/(1+nu)
        self.Hook = np.zeros((9,9))
        I = np.eye(3)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        self.Hook[3*i+j,3*k+l] = self.lambd*I[i,j]*I[k,l] + self.mu*(I[i,k]*I[j,l]+I[i,l]*I[j,k])
    
    def constitutive(self, GP, dispVec, curTime, timeStep, stiff):
        """
        evaluate the constitutive relation at GP

        Parameters
        ----------
        GP : GaussPoint
            all data in a GP.
        dispVec : numpy array
            displacement at this GP.
        curTime : float
            current time.
        timeStep : float
            time step.
        stiff : boolean
            True if tangent needs to be computed.

        Returns
        -------
        None.

        """
        dim = GP.dim        
        defo = np.zeros(9)
        if dim ==2:
            defo[self.locVec2D] = GP.currentState.gradFields[:dim*dim] 
        else:
            defo[:] = GP.currentState.gradFields[:dim*dim] 
        GP.currentState.CauchyStress= np.matmul(self.Hook, defo)
        GP.currentState.defoEnergy = 0.5*np.dot(defo,GP.currentState.CauchyStress)
        pres=(GP.currentState.CauchyStress[0]+GP.currentState.CauchyStress[3*1+1]+GP.currentState.CauchyStress[3*2+2])/3.
        devsig = GP.currentState.CauchyStress.reshape(3,3) - pres*np.eye(3)
        GP.currentState.SVM = np.sqrt(1.5*np.tensordot(devsig,devsig))
        if dim==2:
            GP.currentState.fluxFields[:dim*dim] = GP.currentState.CauchyStress[self.locVec2D]            
            self.computeInternalForce(GP)
            if stiff:
                GP.currentState.tangent[:dim*dim,:dim*dim]= self.Hook[self.locMat2D]
                self.computeStiffness(GP)
        elif dim==3:
            GP.currentState.fluxFields[:dim*dim] = GP.currentState.CauchyStress          
            self.computeInternalForce(GP)
            if stiff:
                GP.currentState.tangent[:dim*dim,:dim*dim]= self.Hook
                self.computeStiffness(GP)
        else:
            raise NotImplementedError("this dimension is not implemented")
            
            
class compressiveNeoHookean(MaterialLaw):
    def __init__(self, lawName, E, nu):
        """
        compressive Neo-Hookean material law

        Parameters
        ----------
        lawName : string
            material name.
        E : float
            Young modulus.
        nu : float
            Poisson ratio.

        Returns
        -------
        None.

        """
        super().__init__()
        self.lawName=lawName
        self.E = E 
        self.nu = nu
        self.lambd = E*nu/(1+nu)/(1-2*nu)
        self.mu = E/2/(1+nu)
        
    def constitutive(self, GP, dispVec, curTime, timeStep, stiff):
        """
        evaluate the constitutive relation at GP

        Parameters
        ----------
        GP : GaussPoint
            all data in a GP.
        dispVec : numpy array
            displacement at this GP.
        curTime : float
            current time.
        timeStep : float
            time step.
        stiff : boolean
            True if tangent needs to be computed.

        Returns
        -------
        None.

        """
        dim = GP.dim
        I = np.eye(3)
        if dim==2:
            F = np.eye(3)
            F[:dim,:dim]+=GP.currentState.gradFields.reshape(dim,dim)
        elif dim==3:
            F = np.eye(3)+GP.currentState.gradFields.reshape(3,3)
        J = np.linalg.det(F)
        C = np.matmul(F.T,F)
        invC = np.linalg.inv(C)
        logJ = np.log(J)
        #PK2=mu*(I-invC)+lambda*(logJ)*invC
        PK2 = self.mu*(I-invC)+self.lambd*logJ*invC
        PK1 = np.matmul(F,PK2)
        GP.currentState.defoEnergy =  0.5* self.lambd*logJ*logJ - self.mu*logJ + 0.5*self.mu*(C[0,0]+C[1,1]+C[2,2]-3)
        GP.currentState.CauchyStress=np.matmul(PK1,F.T)/J
        pres = (GP.currentState.CauchyStress[0,0]+GP.currentState.CauchyStress[1,1]+GP.currentState.CauchyStress[2,2])/3.
        devsig = GP.currentState.CauchyStress - pres*I
        GP.currentState.SVM = np.sqrt(1.5*np.tensordot(devsig,devsig))
        PK1 = PK1.reshape(9)
        GP.currentState.CauchyStress = GP.currentState.CauchyStress.reshape(9)
        if dim == 2:
            GP.currentState.fluxFields[:dim*dim] = PK1[self.locVec2D]   
        elif dim == 3:
            GP.currentState.fluxFields[:dim*dim] = PK1
        self.computeInternalForce(GP)
            
        if stiff:
            GP.currentState.tangent.fill(0)
            DPK2DC=np.zeros((dim*dim,dim*dim))
            for i in range(dim):
                for j in range(dim):
                    for k in range(dim):
                        for l in range(dim):
                            DPK2DC[i*dim+j,k*dim+l] = (self.lambd*logJ - self.mu)*(-0.5)*(invC[i,k]*invC[j,l]+ invC[i,l]*invC[j,k]) + 0.5*self.lambd*invC[i,j]*invC[k,l]
                        
            for i in range(dim):
                for J in range(dim):
                    for k in range(dim):
                        for L in range(dim):
                            for K in range(dim):
                                for I in range(dim):
                                    GP.currentState.tangent[i*dim+J, k*dim+ L] += 2.*DPK2DC[I*dim+J, K*dim+ L]*F[i,I]*F[k,K]
                            if i==k:
                                GP.currentState.tangent[i*dim+ J, k*dim+L] += PK2[J,L]    
            self.computeStiffness(GP)

class J2SmallStrainPlasticity(MaterialLaw):
    def __init__(self, lawName, E, nu, sy0, h):
        """
        compressive Neo-Hookean material law

        Parameters
        ----------
        lawName : string
            material name.
        E : float
            Young modulus.
        nu : float
            Poisson ratio.
        sy0 : float
            initial yield stress.
        h : float
            hardening modulus.

        Returns
        -------
        None.

        """
        super().__init__()
        self.lawName=lawName
        self.E = E 
        self.nu = nu
        self.lambd = E*nu/(1+nu)/(1-2*nu)
        self.mu = E/2/(1+nu)
        self.K = self.lambd+2*self.mu/3.
        self.sy0 = sy0
        self.h = h
        self.Hook=np.zeros((9,9))
        I = np.eye(3)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        self.Hook[i*3+j,k*3+l] = self.lambd*I[i,j]*I[k,l] + self.mu*(I[i,k]*I[j,l]+I[i,l]*I[j,k])
        
        self.Idev=np.zeros((9,9))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        self.Idev[i*3+j,k*3+l] = (I[i,k]*I[j,l]+I[i,l]*I[j,k])*0.5 -I[i,j]*I[k,l]/3. 
    
    def allocateInternalState(self, GP):
        """
        allocate internal state
        Parameters
        ----------
        GP : GaussPoint
            data at GP.

        Returns
        -------
        None.

        """
        # we store [p epspxx epspyy epspzz epspxy epspxz epspyz]
        GP.currentState.internalState=np.zeros(7)
        GP.previousState.internalState=np.zeros(7)
        
        
    def constitutive(self, GP, dispVec, curTime, timeStep, stiff):
        """
        evaluate the constitutive relation at GP

        Parameters
        ----------
        GP : GaussPoint
            all data in a GP.
        dispVec : numpy array
            displacement at this GP.
        curTime : float
            current time.
        timeStep : float
            time step.
        stiff : boolean
            True if tangent needs to be computed.

        Returns
        -------
        None.

        """
        dim = GP.dim
        I = np.eye(3)
        defo = np.zeros((3,3))        
        for i in range(dim):
            for j in range(dim):
                defo[i,j] = 0.5*(GP.currentState.gradFields[i*dim+j]+GP.currentState.gradFields[j*dim+i])
        
        # previous plastic deformation
        GP.currentState.internalState[:] = GP.previousState.internalState
        #
        p0 = GP.currentState.internalState[0]
        epsp0 = np.zeros((3,3))
        epsp0[0,0]=GP.currentState.internalState[1]
        epsp0[1,1]=GP.currentState.internalState[2]
        epsp0[2,2]=GP.currentState.internalState[3]
        epsp0[0,1]=GP.currentState.internalState[4]
        epsp0[0,2]=GP.currentState.internalState[5]
        epsp0[1,2]=GP.currentState.internalState[6]
        epsp0[1,0]=epsp0[0,1]
        epsp0[2,0]=epsp0[0,2]
        epsp0[2,1]=epsp0[1,2]
        
        # predictor
        Ee = defo-epsp0
        traceEe = Ee[0,0]+Ee[1,1]+Ee[2,2]
        devEe = Ee - I*traceEe/3
        pres = self.K*traceEe
        sigDevpr = 2.*self.mu*devEe
        SVMpr= np.sqrt(1.5*np.tensordot(sigDevpr,sigDevpr))
        withPlastic=False
        if SVMpr - self.sy0 - self.h*p0 > 0:
            # plastic occurs
            dp = (SVMpr - self.sy0 - self.h*p0)/(self.h + 3.*self.mu)
            N = 1.5*sigDevpr/SVMpr
            sigDev = sigDevpr - 2*self.mu*dp*N
            SVM = SVMpr - 3.*self.mu*dp
            Ee -= dp*N
            GP.currentState.internalState[0] = p0 + dp
            GP.currentState.internalState[1] = epsp0[0,0]+ dp*N[0,0]
            GP.currentState.internalState[2] = epsp0[1,1]+ dp*N[1,1]
            GP.currentState.internalState[3] = epsp0[2,2]+ dp*N[2,2]
            GP.currentState.internalState[4] = epsp0[0,1]+ dp*N[0,1]
            GP.currentState.internalState[5] = epsp0[0,2]+ dp*N[0,2]
            GP.currentState.internalState[6] = epsp0[1,2]+ dp*N[1,2]
            withPlastic=True
        else:
            sigDev = sigDevpr
            SVM = SVMpr
        GP.currentState.CauchyStress = sigDev.reshape(9) + pres*I.reshape(9)
        GP.currentState.defoEnergy = 0.5*np.dot(GP.currentState.CauchyStress,Ee.reshape(9))     
        GP.currentState.SVM = SVM
        if dim ==2:
            GP.currentState.fluxFields[:dim*dim] = GP.currentState.CauchyStress[self.locVec2D]
            self.computeInternalForce(GP)
            if stiff:
                GP.currentState.tangent = self.Hook[self.locMat2D]
                if withPlastic:
                    DDeltaDeps = self.Idev[self.locMat2D]*(dp*3.*self.mu)/SVMpr
                    for i in range(dim):
                        for j in range(dim):
                            for k in range(dim):
                                for l in range(dim):
                                    DDeltaDeps[i*dim+j,k*dim+l] += (2*self.mu/(self.h+3.*self.mu) - (dp*2.*self.mu)/SVMpr)*N[i,j]*N[k,l]
                    DDeltaDeps *= (-2.*self.mu)
                    GP.currentState.tangent += DDeltaDeps
                self.computeStiffness(GP)
        elif dim==3:
            GP.currentState.fluxFields[:dim*dim] = GP.currentState.CauchyStress
            self.computeInternalForce(GP)
            if stiff:
                GP.currentState.tangent[:]=self.Hook
                if withPlastic:
                    DDeltaDeps = self.Idev*(dp*3.*self.mu)/SVMpr
                    for i in range(dim):
                        for j in range(dim):
                            for k in range(dim):
                                for l in range(dim):
                                    DDeltaDeps[i*dim+j,k*dim+l] += (2*self.mu/(self.h+3.*self.mu) - (dp*2.*self.mu)/SVMpr)*N[i,j]*N[k,l]
                    DDeltaDeps *= (-2.*self.mu)   
                    GP.currentState.tangent += DDeltaDeps
                self.computeStiffness(GP)
        else:
            raise NotImplementedError("this dimension is not implemented")
                
def createLaw(lawName, lawType, parameters):
      """
      create material law based on material type
      """
      print(f"create material law: {locals()}")
      if lawType == "LINEAR-ELASTIC":
          return  LinearElasticLaw(lawName,parameters[0],parameters[1])
      elif lawType == "NEO-HOOKEAN":
          return compressiveNeoHookean(lawName,parameters[0],parameters[1])
      elif lawType == "J2PLASTIC":
          return J2SmallStrainPlasticity(lawName,parameters[0],parameters[1],parameters[2],parameters[3])
      else:
        raise NotImplementedError("lawName has not been implemented for material law")
    
    
if __name__ == "__main__":
    a = createLaw("myLaw","LINEAR-ELASTIC",[1e9,0.3])
  
