# Element class
#
import numpy as np
import Node

class QuadratureRule:
    def __init__(self):
        pass
    @staticmethod
    def getDefaultOrder(elementType):
        if elementType == "QUAD4":
            return 2
        elif elementType=="TRI3":
            return 1
        elif elementType=="TRI6":
            return 4
        elif elementType=="TET4":
            return 1
        elif elementType=="TET10":
            return 2
        else:
            raise NotImplementedError("no default order is defined for this element")
    
    @staticmethod
    def getGPs(elementType, order):
        if elementType=="QUAD4":
            return QuadratureRule.getGPs_QUAD(order)
        elif elementType=="TRI3" or elementType=="TRI6":
            return QuadratureRule.getGPs_TRI(order)
        elif elementType =="TET4" or elementType=="TET10":
            return QuadratureRule().getGPs_TET(order)
        else:
            raise NotImplementedError("no default order is defined for this element")
    
    @staticmethod
    def getGPs_TET(order):
        """
        get Gausspoints in tetrahedral elements
        
        Parameters
        ----------
        order : int
            order of integration.

        Raises
        ------
        NotImplementedError
            order is not defined.

        Returns
        -------
        npts : int
            Number of GPS.
        coords_gps : numpy array of GPs (npts, dim)
            Coordinates of GPs.
        weights : numpy array
            weigt at each GP.

        """
        if order == 1:
            npts=1
            coords_gps = np.array([[0.25,0.25,0.25]])
            weights = np.array([0.166666666667])
            return npts, coords_gps, weights
        elif order ==2:
            npts=4
            coords_gps = np.zeros((4,3))
            weights =  np.zeros(4)
            coords_gps[0][0] = (5. + 3. * np.sqrt(5.)) / 20.;
            coords_gps[0][1] = (5. - np.sqrt(5.)) / 20.;
            coords_gps[0][2] = (5. - np.sqrt(5.)) / 20.;
            weights[0] = 1. / 24.;

            coords_gps[1][0] = (5. - np.sqrt(5.)) / 20.;
            coords_gps[1][1] = (5. + 3. * np.sqrt(5.)) / 20.;
            coords_gps[1][2] = (5. - np.sqrt(5.)) / 20.;
            weights[1] = 1. / 24.;

            coords_gps[2][0] = (5. - np.sqrt(5.)) / 20.;
            coords_gps[2][1] = (5. - np.sqrt(5.)) / 20.;
            coords_gps[2][2] = (5. + 3. * np.sqrt(5.)) / 20.;
            weights[2]= 1. / 24.;

            coords_gps[3][0] = (5. -np. sqrt(5.)) / 20.;
            coords_gps[3][1] = (5. - np.sqrt(5.)) / 20.;
            coords_gps[3][2] = (5. - np.sqrt(5.)) / 20.;
            weights[3]= 1. / 24.;
            return npts, coords_gps, weights
        else:
            raise NotImplementedError("GP is not implemented for this type of elemet")
            
    @staticmethod
    def getGPs_QUAD(order):
        """
        get Gausspoints in quadrangular elements
        
        Parameters
        ----------
        order : int
            order of integration.

        Raises
        ------
        NotImplementedError
            order is not defined.

        Returns
        -------
        npts : int
            Number of GPS.
        coords_gps : numpy array of GPs (npts, dim)
            Coordinates of GPs.
        weights : numpy array
            weigt at each GP.

        """
        if order == 1:
            npts=1
            coords_gps = np.array([[0.,0.]])
            weights = np.array([4.])
            return npts, coords_gps, weights
        elif order == 2:
            npts=4
            coords_gps = np.zeros((4,2))
            weights =  np.zeros(4)
            coords_gps[0][0] = 0.577350269
            coords_gps[0][1] = 0.577350269
            weights[0] = 1
            coords_gps[1][0] = -0.577350269
            coords_gps[1][1] = 0.577350269
            weights[1] = 1
            coords_gps[2][0] = 0.577350269
            coords_gps[2][1] = -0.577350269
            weights[2] = 1
            coords_gps[3][0] = -0.577350269
            coords_gps[3][1] = -0.577350269
            weights[3] = 1
            return npts, coords_gps, weights
        else:
            raise NotImplementedError("GP is not implemented for this type of elemet")
            
    @staticmethod
    def getGPs_TRI(order):
        """
        get Gausspoints in triangle elements
        
        Parameters
        ----------
        order : int
            order of integration.

        Raises
        ------
        NotImplementedError
            order is not defined.

        Returns
        -------
        npts : int
            Number of GPS.
        coords_gps : numpy array of GPs (npts, dim)
            Coordinates of GPs.
        weights : numpy array
            weigt at each GP.

        """
        if order == 1:
            npts=1
            coords_gps = np.array([[0.333333333333333, 0.333333333333333]])
            weights = np.array([0.5])
            return npts, coords_gps, weights
        elif order == 2:
            npts=3
            coords_gps = np.zeros((npts,2))
            weights =  np.zeros(npts)
            coords_gps[0][0] = 0.166666666666667
            coords_gps[0][1] = 0.166666666666667
            weights[0] = 0.166666666666667
            coords_gps[1][0] = 0.666666666666667
            coords_gps[1][1] = 0.666666666666667
            weights[1] = 0.166666666666667
            coords_gps[2][0] = 0.166666666666667
            coords_gps[2][1] = 0.166666666666667
            weights[2] = 0.166666666666667
            return npts, coords_gps, weights
        elif order == 4:
            npts=6
            coords_gps = np.zeros((npts,2))
            weights =  np.zeros(npts)
            coords_gps[0][0] = 0.44594849091597
            coords_gps[0][1] = 0.44594849091597
            weights[0] = 0.111690794839005

            coords_gps[1][0] = 0.44594849091597
            coords_gps[1][1] = 0.10810301816807
            weights[1] = 0.111690794839005

            coords_gps[2][0] = 0.10810301816807
            coords_gps[2][1] = 0.44594849091597
            weights[2] = 0.111690794839005

            coords_gps[3][0] = 0.09157621350977
            coords_gps[3][1] = 0.09157621350977
            weights[3] = 0.054975871827661

            coords_gps[4][0] = 0.09157621350977
            coords_gps[4][1] = 0.81684757298046
            weights[4] = 0.054975871827661

            coords_gps[5][0] = 0.81684757298046
            coords_gps[5][1] = 0.09157621350977
            weights[5] = 0.054975871827661
            return npts, coords_gps, weights
        else:
            raise NotImplementedError("GP is not implemented for this type of elemet")

class IsoParametricShapeFunction:
    def __init__(self, elementType):
        self.elementType = elementType
    
    def shapeFunctions(self, uvw):
        phi=None
        dphiIso=None
        u = uvw[0]
        v = uvw[1]
        if self.elementType == "TRI3":
            phi = np.zeros(3)
            phi[0]= 1-u-v
            phi[1]= u
            phi[2]= v
            
            dphiIso = np.zeros((3,2))
            dphiIso[0][0]=-1.
            dphiIso[0][1]=-1.
            dphiIso[1][0]=1.
            dphiIso[1][1]=0.
            dphiIso[2][0]=0.
            dphiIso[2][1]=1.
        elif self.elementType == "TRI6":
            phi=np.zeros(6)
            phi[0]= (1-u-v)*(1-2*u-2*v)
            phi[1]= u*(2*u-1)
            phi[2]= v*(2*v-1)
            phi[3]= 4*(1-u-v)*u
            phi[4]= 4*u*v
            phi[5]= 4*(1-u-v)*v
            
            dphiIso = np.zeros((6,2))
            dphiIso[0][0]=-(1-2*u-2*v)-2*(1-u-v)
            dphiIso[0][1]=-(1-2*u-2*v)-2*(1-u-v)
            dphiIso[1][0]=4*u-1
            dphiIso[1][1]=0.
            dphiIso[2][0]=0
            dphiIso[2][1]=4*v-1
            dphiIso[3][0]=4*(1-u-v)-4*u
            dphiIso[3][1]=-4*u
            dphiIso[4][0]= 4*v
            dphiIso[4][1]= 4*u
            dphiIso[5][0]= -4*v
            dphiIso[5][1]= 4*(1-u-v)-4*v
        elif self.elementType=="QUAD4":
            phi = np.zeros(4)
            phi[0]=0.25 * (1 - u) * (1 - v)
            phi[1]=0.25 * (1 + u) * (1 - v)
            phi[2]=0.25 * (1 + u) * (1 + v)
            phi[3]=0.25 * (1 - u) * (1 + v)
            
            dphiIso = np.zeros((4,2))
            dphiIso[0][0]=-0.25*(1-v)
            dphiIso[0][1]=-0.25*(1-u)
            dphiIso[1][0]=0.25*(1-v)
            dphiIso[1][1]=-0.25*(1+u)
            dphiIso[2][0]=0.25*(1+v)
            dphiIso[2][1]=0.25*(1+u)
            dphiIso[3][0]=-0.25*(1+v)
            dphiIso[3][1]=0.25*(1-u)
        
        elif self.elementType=="TET4":
            w=uvw[2]
            phi = np.zeros(4)
            phi[0]=1-u-v-w
            phi[1]=u
            phi[2]=v
            phi[3]=w
            
            dphiIso = np.zeros((4,3))
            dphiIso[0][0]=-1
            dphiIso[0][1]=-1
            dphiIso[0][2]=-1
            dphiIso[1][0]=1
            dphiIso[1][1]=0
            dphiIso[1][2]=0
            dphiIso[2][0]=0
            dphiIso[2][1]=1
            dphiIso[2][2]=0
            dphiIso[3][0]=0
            dphiIso[3][1]=0
            dphiIso[3][2]=1
        elif self.elementType=="TET10":
            w = uvw[2]
            phi=np.zeros(10)
            #
            t1 = 1-u-v-w
            t2 = u
            t3 = v
            t4 = w
            # 
            phi[0]=t1*(2*t1-1)
            phi[1]=t2*(2*t2-1)
            phi[2]=t3*(2*t3-1)
            phi[3]=t4*(2*t4-1)
            phi[4]=4*t1*t2
            phi[5]=4*t2*t3
            phi[6]=4*t3*t1
            phi[7]=4*t1*t4
            phi[8]=4*t3*t4
            phi[9]=4*t2*t4
            
            dphiIso = np.zeros((10,3))
            dphiIso[0][0]=-4*t1+1
            dphiIso[0][1]=-4*t1+1
            dphiIso[0][2]=-4*t1+1
            #
            dphiIso[1][0]=4*t2-1
            dphiIso[1][1]=0
            dphiIso[1][2]=0
            
            dphiIso[2][0]=0
            dphiIso[2][1]=4*t3-1
            dphiIso[2][2]=0
            
            dphiIso[3][0]=0
            dphiIso[3][1]=0
            dphiIso[3][2]=4*t4-1
            
            dphiIso[4][0]=-4*t2+4*t1
            dphiIso[4][1]=-4*t2
            dphiIso[4][2]=-4*t2
            
            dphiIso[5][0]=4*t3
            dphiIso[5][1]=4*t2
            dphiIso[5][2]=0
            
            dphiIso[6][0]=-4*t3
            dphiIso[6][1]=-4*t3+4*t1
            dphiIso[6][2]=-4*t3
                  
            dphiIso[7][0]=-4*t4
            dphiIso[7][1]=-4*t4
            dphiIso[7][2]=-4*t4+4*t1
            
            dphiIso[8][0]=0
            dphiIso[8][1]=4*t4
            dphiIso[8][2]=4*t3
            
            dphiIso[9][0]=4*t4
            dphiIso[9][1]=0
            dphiIso[9][2]=4*t2
        else:
            raise NotImplementedError("this kind of element is not implemented")
            
        return phi, dphiIso
    
    
class Element:
    """ general class for elements
    """
    def __init__(self, index, nodeList, elementType, matLawIndex):
        """
        Parameters
        ----------
        index : int
            identification number.
        nodeList : list of nodes
            list of nodes of this element.
        elementType : string
            type element.
        matLawIndex : int
            material law index.

        Returns
        -------
        None.

        """
        self.index=index
        self.nodeList=nodeList
        self.nodeListIndices= [n.index for n in nodeList]
        self.matLawIndex = matLawIndex
        self.elementType=elementType
        self.GPPositionStart=0
        self.GPPositionEnd=0
        self.numNodes=len(self.nodeList)
        if self.numNodes==0:
            raise RuntimeError("wrong element initialisation")
        self.dim= self.nodeList[0].dim
        self.isoSpace = IsoParametricShapeFunction(self.elementType)
        
    def createGaussPoints(self, order, GPs):
        """
        generate the gauss points

        Parameters
        ----------
        order : int
            order of integration.

        Returns
        -------
        GPs : TYPE
            list of GaussPoint.

        """
        if order >0:
            num_gpts, coordinates_Gps, weights = QuadratureRule.getGPs(self.elementType,order)
        else:
            num_gpts, coordinates_Gps, weights = QuadratureRule.getGPs(self.elementType,QuadratureRule.getDefaultOrder(self.elementType))
        self.GPPositionStart= len(GPs)
        for igp in range(num_gpts):
            uvw = coordinates_Gps[igp]
            phi, dphiIso = self.isoSpace.shapeFunctions(uvw)
            # coordinates of 
            coords = np.zeros(self.dim)
            for i in range(self.numNodes):
                for j in range(self.dim):
                    coords[j]+= phi[i]*self.nodeList[i].coords[j]
            # append new GP
            GPs.append(Node.GaussPoint(igp, coords=coords, neighbors=self.nodeListIndices, elementIndex=self.index))
            # Jacobian
            J = np.zeros((self.dim, self.dim))
            for i in range(self.numNodes):
                for j in range(self.dim):
                    for k in range(self.dim):
                        J[j,k] += dphiIso[i,k]*self.nodeList[i].coords[j]
            # gradient of shape function
            detJ = np.linalg.det(J)
            invJ = np.linalg.inv(J)
            # numdof
            numDofsPerNode = GPs[-1].numFields
            GPs[-1].shapeFunc[:]=0.
            for j in range(numDofsPerNode):
                GPs[-1].shapeFunc[j,j*self.numNodes:(j+1)*self.numNodes] = phi
            
            dphiXYZ = np.zeros((self.numNodes,self.dim))
            for i in range(self.numNodes):
                dphiXYZ[i] = np.matmul(dphiIso[i],invJ)
                        
            GPs[-1].shapeFuncGrad[:,:]=0
            for j in range(numDofsPerNode):
                for k in range(self.dim):
                    pos = j*self.dim+k
                    GPs[-1].shapeFuncGrad[pos, j*self.numNodes:(j+1)*self.numNodes] =  dphiXYZ[:,k]
            # volume fraction
            GPs[-1].wJ = np.abs(detJ*weights[igp])           
        self.GPPositionEnd= self.GPPositionStart + num_gpts
    
    def __str__(self):
        return (f"element {self.index} \nmaterial law {self.matLawIndex} \nelement type {self.elementType}\n node list {self.nodeListIndices}\n")
        
if __name__ == "__main__":
  
    npts, coords, weights = QuadratureRule.getGPs_QUAD(2)
    print(npts, coords, weights)
