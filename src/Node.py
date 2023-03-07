
import numpy as np

class Node:
    def __init__(self, index, coords, nodeType):
        """
        constructor of Node

        Parameters
        ----------
        index : int
            Unique identification number of node.
        coords : numpy array
            coordinate.
        nodeType : string
            node type name.

        Returns
        -------
        None.

        """
        self.index=index
        self.coords=coords
        self.dim = len(coords)
        self.nodeType=nodeType
        self.numFields=self.dim
        

class State:
    def __init__(self, totalDofs, numFields, dim):
        """
        state at each Gauss point

        Parameters
        ----------
        dim : int
            dimension of the problem.
        totalDofs : int
            total number of degree of freedom associated to this state.

        Returns
        -------
        None.

        """
        # strain measure
        self.gradFields=np.zeros(numFields*dim)
        # stress measure
        self.fluxFields=np.zeros(numFields*dim)
        # derivatives of stress versus deformation
        self.tangent=np.zeros((numFields*dim,numFields*dim))
        # internal force
        self.internalForce=np.zeros(totalDofs)
        # external force
        self.externalForce=np.zeros(totalDofs)
        # stiffness = D(internalForce-externalForce)/D-dofs 
        self.stiffness=np.zeros((totalDofs,totalDofs))
        # internal state
        self.internalState = None
        # defo energy
        self.defoEnergy = 0.
        # vonmise stress
        self.SVM= 0.
        # cauchy stress
        self.CauchyStress=np.zeros(9)
  
    def setData(self, source):
        """
        copy from other data

        Parameters
        ----------
        source : State
            other state.

        Returns
        -------
        None.

        """
        # deformation measure
        self.gradFields[:]=(source.gradFields)
        # stress measure
        self.fluxFields[:]=(source.fluxFields)
        # derivatives of stress versus deformation
        self.tangent[:]=(source.tangent)
        # internal force
        self.internalForce[:]=(source.internalForce)
        # external force
        self.externalForce[:]=(source.externalForce)
        # stiffness = D(internalForce-externalForce)/D-dofs 
        self.stiffness[:]=(source.stiffness)
        # internal state
        if (self.internalState is not None) and (source.internalState is not None):
            self.internalState[:]=(source.internalState)
        # defo energy
        self.defoEnergy = source.defoEnergy
        # vonmise stress
        self.SVM=source.SVM
        # cauchy stress
        self.CauchyStress[:]=source.CauchyStress
        
class GaussPoint(Node):
    def __init__(self, index, coords, neighbors, elementIndex):
        """
        Constructor 

        Parameters
        ----------
        index : int
            position of the GP in the element.
        coords : numpy array
            coordinate.
        neighbors : list of int
            neighbouring nodes.
        elementIndex : int
            Element index this GP belongs to.

        Returns
        -------
        None.

        """
        super().__init__(index,coords,"GaussPoint")
        self.neighbors= neighbors
        self.totalDofs=len(neighbors)*self.numFields
        # shape function
        self.shapeFunc= np.zeros((self.numFields,self.numFields*len(neighbors)))
        # gradient of shape functions
        self.shapeFuncGrad= np.zeros((self.numFields*self.dim,self.numFields*len(neighbors)))
        # current state
        self.currentState = State(self.totalDofs, self.numFields, self.dim)
        # previous step
        self.previousState = State(self.totalDofs, self.numFields, self.dim)
        # local dof
        self.localDofs=[]
        for j in range(self.numFields):
            for nn in neighbors:
                self.localDofs.append((nn,j))
        self.elementIndex=elementIndex
        self.wJ=0;
            
    def nextStep(self):
        """
        copy to next step

        Returns
        -------
        None.

        """
        self.previousState.setData(self.currentState)
    
    
    def resetToPreviousStep(self):
        """
        reset to previous step

        Returns
        -------
        None.

        """
        self.currentState.setData(self.previousState)
        
    def __str__(self):
        return f"GP {self.index} at element {self.elementIndex} coords={self.coords}\n"
