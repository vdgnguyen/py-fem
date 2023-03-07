
import numpy as np
import System
import Node
import Element
import MaterialLaw as law
import time

class DirichletBC:
    def __init__(self, value):
        """
        Parameters
        ----------
        value : float
            prescribed displacement value.

        Returns
        -------
        None.

        """
        self.value=value
    
    def getValue(self, curTime):
        """
        get value as a linear function of time

        Parameters
        ----------
        curTime : float
            current time.

        Returns
        -------
        float
            current prescribed displacement.

        """
        return self.value*curTime
    
    def __str__(self):
        return f"DirichletBC value = {self.value}"
    
class saveData:
    def __init__(self, dataType, comp, saveFileName, supportList):
        print(f"allocate saveData {locals()}")
        self.dataType=dataType
        self.comp = comp
        self.saveFileName = saveFileName
        self.supportList=supportList
        self.filePt = None
        
    def openFile(self, prefix):
        self.filePt = open(prefix+self.saveFileName,"w")
        self.filePt.write("Time,Value\n")
        self.filePt.flush()
        self.lastTime =0
        
    def writeData(self, curTime, solver):
        if self.lastTime < curTime:
            self.lastTime = curTime
            if self.dataType == "INTERNAL-FORCE":
                val = 0
                for nn in self.supportList:
                    dof = nn -1 + self.comp*solver.numNodes
                    val += solver.system.FintFull[dof]
                self.filePt.write(f"{curTime},{val}\n")
                self.filePt.flush()
            else:
                raise NotImplementedError("data type is not been implemented")
        
    def closeFile(self):
        self.filePt.close()

class FEMSolver:
    def __init__(self, name="myTest"):
        """
        initial FEM solver
        """
        self.system = None # system to solve
        self.nodes={} # dict of list of floats vectors {i: [x0,x1,...]}
        self.elements={} # dict of list int {i: [node0, node1, ...]}
        self.fixedDofs={} # dict of pair {(nodeIndex,comp): value}
        self.materialLaws={} # all material law
        self.GPs=[] # list of Gauss points
        self.dofDict={} # dofDicts
        self.numNodes = 0
        self.numElements=0
        self.dataArchiving=[]
        self.name = name
    
    def loadMeshFile(self, meshFileName):
        """
        read mesh file name
        Args:
          meshFileName (string): mesh file name
        """
        with open(meshFileName) as fp:
            Lines = fp.readlines()
            iline=-1
            # read nodes
            while iline < len(Lines)-1:
                iline+=1
                #print(f"fffffffff line {iline} {Lines[iline]}")
                if "*NODE" in Lines[iline]:
                    while True:
                        iline += 1
                        #print(f"line {iline} {Lines[iline]}")
                        if "*END" in Lines[iline]:
                            break 
                        line = Lines[iline]
                        line = line.split(",")
                        lineStrip = [a.strip() for a in line]
                        nodeIndex=int(lineStrip[0])
                        coords = [float(a) for a in lineStrip[1:]]
                        self.nodes[nodeIndex] = Node.Node(nodeIndex,coords,"MeshVertex")
      
                # read elements
                if "*ELEMENT" in Lines[iline]:
                    strTemp = Lines[iline].split("TYPE=")
                    eleType= strTemp[-1].strip()
                    while True:
                        iline += 1
                        #print(f"line {iline} {Lines[iline]}")
                        if "*END" in Lines[iline]:
                            break 
                        line = Lines[iline]
                        line = line.split(",")
                        lineStrip = [a.strip() for a in line]
                        aa = [int(a) for a in lineStrip]
                        nodeList = [self.nodes[int(a)] for a in aa[1:] ]
                        self.elements[aa[0]] = Element.Element(aa[0],nodeList,eleType,-1) # material is initialsed by -1 as it is not initialised yet

                # read material law
                if "*MATERIAL" in Lines[iline]: 
                    matName="None"
                    matType="None"
                    parameters=[]
                    supportEle=[]
                    while True:
                        iline+=1
                        #print(f"line {iline} {Lines[iline]}")
                        if "*END" in Lines[iline]:
                            break
                    
                        if Lines[iline].strip() ==  "NAME":
                            iline+=1
                            #print(f"line {iline} {Lines[iline]}")
                            matName = Lines[iline].strip()

                        if Lines[iline].strip() ==  "TYPE":
                            iline+=1
                            #print(f"line {iline} {Lines[iline]}")
                            matType = Lines[iline].strip()

                        if Lines[iline].strip() ==  "PARAMETERS":
                            iline+=1
                            #print(f"line {iline} {Lines[iline]}")
                            line = Lines[iline].split(",")
                            parameters=[float(a.strip()) for a in line]
                    
                        if Lines[iline].strip() ==  "SUPPORT":
                            iline+=1
                            #print(f"line {iline} {Lines[iline]}")
                            if Lines[iline].strip() == "ALL":
                                for key in self.elements.keys():
                                    supportEle.append(key)
                            else:
                                while True:
                                    eleIndex = int(Lines[iline].strip())
                                    supportEle.append(eleIndex)
                                    iline += 1
                                    #print(f"line {iline} {Lines[iline]}")
                                    if "*END" in Lines[iline]:
                                        iline -= 1
                                        break 
                                
                    lawIndex = len(self.materialLaws)+1
                    #print("create law")
                    self.materialLaws[lawIndex] = law.createLaw(matName,matType,parameters)
                    for ele in supportEle:
                        self.elements[ele].matLawIndex=lawIndex
                
                # read dirichlet BC
                if "*DIRICHLET" in Lines[iline]: 
                    while True:
                        iline+=1
                        #print(f"line {iline} {Lines[iline]}")
                        if "*END" in Lines[iline]:
                            break
                        
                        line = Lines[iline].split(",")
                        line=[(a.strip()) for a in line]
                        self.fixedDofs[(int(line[0]),int(line[1]))] = DirichletBC(float(line[2]))
                        
                # read dirichlet BC
                if "*EXTRACT-NODE" in Lines[iline]: 
                    saveFile="None"
                    saveType="None"
                    comp = -1
                    supportList=[]
                    while True:
                        iline+=1
                        #print(f"line {iline} {Lines[iline]}")
                        if "*END" in Lines[iline]:
                            break
                    
                        if Lines[iline].strip() ==  "SAVEFILE":
                            iline+=1
                            #print(f"line {iline} {Lines[iline]}")
                            saveFile = Lines[iline].strip()

                        if Lines[iline].strip() ==  "TYPE":
                            iline+=1
                            #print(f"line {iline} {Lines[iline]}")
                            saveType = Lines[iline].strip()

                        if Lines[iline].strip() ==  "COMPONENT":
                            iline+=1
                            #print(f"line {iline} {Lines[iline]}")
                            comp = int(Lines[iline].strip())
                    
                        if Lines[iline].strip() ==  "SUPPORT":
                            iline+=1
                            #print(f"line {iline} {Lines[iline]}")
                        
                            while True:
                                eleIndex = int(Lines[iline].strip())
                                supportList.append(eleIndex)
                                iline += 1
                                #print(f"line {iline} {Lines[iline]}")
                                if "*END" in Lines[iline]:
                                    iline -= 1
                                    break 
                    self.dataArchiving.append(saveData(saveType, comp, saveFile, supportList))
                    
        print("done reading input file")
        print("create Gauss Points")
        for key, ele in self.elements.items():
            ele.createGaussPoints(-1,self.GPs)
            for igp in range(ele.GPPositionStart, ele.GPPositionEnd):
                self.materialLaws[ele.matLawIndex].allocateInternalState(self.GPs[igp])
        
        for key, node in self.nodes.items():
            for j in range(node.dim):
                if ((node.index, j) not in self.dofDict.keys()) and ((node.index, j)  not in self.fixedDofs.keys()):
                    size = len(self.dofDict)
                    self.dofDict[(node.index, j)] = size 
        
        self.numNodes = len(self.nodes)
        self.numElements =len(self.elements)
        
    def getUnknown(self, curTime, nodeIndex, comp):
        """
        get solution at a node and component
        Args:
          nodeIndex (int): node index
          comp (int): displacement component
        Returns:
           float
        """  
        if (nodeIndex, comp) in self.fixedDofs.keys():
            return self.fixedDofs[(nodeIndex, comp)].getValue(curTime)
        else:
            dofIndex = self.dofDict[(nodeIndex, comp)]
        return self.system.getSolution(dofIndex)
  
    def getUnknownNodeList(self, curTime, nodeList, compList):
        """
        get solution at a node and component
        Args:
          nodeList (list of int): node indices
          comp (list of int): displacement components
        Returns:
           np.array of float
        """  
        nNodes=len(nodeList)
        nComps=len(compList)
        res= np.zeros(nNodes*nComps)
        for a in range(nNodes):
            for i in range(nComps):
                res[a+ nNodes*i] = self.getUnknown(curTime,nodeList[a],compList[i])
        return res
  
    def computeIPVariable(self, curTime, timeStep, stiff):
        """
        compute state at each Gauss point

        Parameters
        ----------
        curTime : float
            current time.
        timeStep : float
            current time step.
        stiff : boolean
            true if stiffness is needed.

        Returns
        -------
        None.

        """
        # compute deformation gradient
        start_time = time.time()
        allComps=[i for i in range(self.GPs[0].dim)]
        for gp in self.GPs:
            # get unknown vector at GP
            dispVec = self.getUnknownNodeList(curTime,gp.neighbors,allComps)
            # compute grad fields
            gp.currentState.gradFields = np.matmul(gp.shapeFuncGrad,dispVec)
            # evaluate constitutive law at GP
            element = self.elements[gp.elementIndex]
            self.materialLaws[element.matLawIndex].constitutive(gp,dispVec,curTime, timeStep, stiff)
        print("solving time in computeIPVariable: %e seconds" % (time.time() - start_time))
        
    def computeRightHandSide(self, curTime):
        """
        compute right hand side of the nonlinear system

        Returns
        -------
        None.

        """
        # zero RHS first
        start_time = time.time()
        self.system.zeroRHS()
        for gp in self.GPs:
            for j in range(gp.totalDofs):
                pos = gp.localDofs[j][0]-1 + self.numNodes*gp.localDofs[j][1]
                self.system.addToFintFull(pos,gp.currentState.internalForce[j])
                if gp.localDofs[j] not in self.fixedDofs:
                    row= self.dofDict[gp.localDofs[j]]
                    self.system.addToFint(row, gp.currentState.internalForce[j])
        print("solving time in computeRightHandSide: %e seconds" % (time.time() - start_time))
        
    def computeStiffness(self):
        """
        compute stiffness of the system

        Returns
        -------
        None.

        """
        # zero RHS first
        start_time = time.time()
        self.system.zeroMatrix()
        for gp in self.GPs:
            for j in range(gp.totalDofs):
                if gp.localDofs[j] not in self.fixedDofs:
                    row= self.dofDict[gp.localDofs[j]]
                    for k in range(gp.totalDofs):
                        if gp.localDofs[k] not in self.fixedDofs:
                            col= self.dofDict[gp.localDofs[k]]
                            self.system.addToMatrix(row, col, gp.currentState.stiffness[j,k])
        print("solving time in computeStiffness: %e seconds" % (time.time() - start_time))
    
    def achiving(self, curTime):
        for dataSave in self.dataArchiving:
            if dataSave.filePt is None:
                dataSave.openFile(self.name+"-")
            dataSave.writeData(curTime,self)
            
    def computeTotalDeformationEnergy(self):
        """
        compute total energy
        Returns
        -------
        energ : float
            total energy of the system.

        """
        energ=0.
        for GP in self.GPs:
            energ += GP.wJ*GP.currentState.defoEnergy
        return energ
    
    def writeData(self, curTime, step):
        """
        write displacement and stress data to file for visualisation

        Parameters
        ----------
        curTime : float
            current time.
        step : float
            current step index.
        fileNamePrefix : string
            prefix to save file.

        Returns
        -------
        None.

        """
        start_time = time.time()
        outFile = open(self.name+f"-disp-{step}.msh","w")
        outFile.write("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"); 
        outFile.write("$NodeData\n");
        outFile.write(f"1\n\"displacement\"\n1\n{curTime}\n3\n{step}\n{3}\n{self.numNodes}\n");
        for key, nn in self.nodes.items():
            disp = np.zeros(3)
            for i in range(nn.dim):
                disp[i] = self.getUnknown(curTime,nn.index,i)
            outFile.write(f"{nn.index} {disp[0]} {disp[1]} {disp[2]}\n")
        outFile.write("$EndNodeData");
        outFile.close()
        
        dim = self.GPs[0].dim
        outFile = open(self.name+f"-stress-{step}.msh","w")
        outFile.write("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"); 
       
        for i in range(dim):
            for j in range(dim):
                outFile.write("$ElementData\n");
                outFile.write(f"1\n\"P{i}{j}\"\n1\n{curTime}\n3\n{step}\n{1}\n{self.numElements}\n");
                for key, ele in self.elements.items():
                    value = 0.
                    volume = 0.
                    for igp in range(ele.GPPositionStart,ele.GPPositionEnd):
                        value += self.GPs[igp].wJ*self.GPs[igp].currentState.fluxFields[dim*i+j]
                        volume += self.GPs[igp].wJ
                    outFile.write(f"{ele.index} {value/volume}\n")
                outFile.write("$EndElementData\n");
                
        for i in range(dim):
            for j in range(dim):
                outFile.write("$ElementData\n");
                outFile.write(f"1\n\"F{i}{j}\"\n1\n{curTime}\n3\n{step}\n{1}\n{self.numElements}\n");
                for key, ele in self.elements.items():
                    value = 0.
                    volume = 0.
                    for igp in range(ele.GPPositionStart,ele.GPPositionEnd):
                        value += self.GPs[igp].wJ*self.GPs[igp].currentState.gradFields[dim*i+j]
                        volume += self.GPs[igp].wJ
                    outFile.write(f"{ele.index} {value/volume}\n")
                outFile.write("$EndElementData\n");
        for i in range(dim):
            for j in range(i,dim):
                outFile.write("$ElementData\n");
                outFile.write(f"1\n\"SIG{i}{j}\"\n1\n{curTime}\n3\n{step}\n{1}\n{self.numElements}\n");
                for key, ele in self.elements.items():
                    value = 0.
                    volume = 0.
                    for igp in range(ele.GPPositionStart,ele.GPPositionEnd):
                        value += self.GPs[igp].wJ*self.GPs[igp].currentState.CauchyStress[3*i+j]
                        volume += self.GPs[igp].wJ
                    outFile.write(f"{ele.index} {value/volume}\n")
                outFile.write("$EndElementData\n");
        outFile.write("$ElementData\n");
        outFile.write(f"1\n\"Deformation Energy\"\n1\n{curTime}\n3\n{step}\n{1}\n{self.numElements}\n");
        for key, ele in self.elements.items():
            value = 0.
            volume = 0.
            for igp in range(ele.GPPositionStart,ele.GPPositionEnd):
                value += self.GPs[igp].wJ*self.GPs[igp].currentState.defoEnergy
                volume += self.GPs[igp].wJ
            outFile.write(f"{ele.index} {value/volume}\n")
        outFile.write("$EndElementData\n");
        
        outFile.write("$ElementData\n");
        outFile.write(f"1\n\"SVM\"\n1\n{curTime}\n3\n{step}\n{1}\n{self.numElements}\n");
        for key, ele in self.elements.items():
            value = 0.
            volume = 0.
            for igp in range(ele.GPPositionStart,ele.GPPositionEnd):
                value += self.GPs[igp].wJ*self.GPs[igp].currentState.SVM
                volume += self.GPs[igp].wJ
            outFile.write(f"{ele.index} {value/volume}\n")
        outFile.write("$EndElementData\n");
        
        if self.GPs[0].currentState.internalState is not None:
            numInVars = len(self.GPs[0].currentState.internalState)
            for ii in range(numInVars):
                outFile.write("$ElementData\n");
                outFile.write(f"1\n\"INVAR{ii}\"\n1\n{curTime}\n3\n{step}\n{1}\n{self.numElements}\n");
                for key, ele in self.elements.items():
                    value = 0.
                    volume = 0.
                    for igp in range(ele.GPPositionStart,ele.GPPositionEnd):
                        value += self.GPs[igp].wJ*self.GPs[igp].currentState.internalState[ii]
                        volume += self.GPs[igp].wJ
                    outFile.write(f"{ele.index} {value/volume}\n")
                outFile.write("$EndElementData\n");
                        
        outFile.close()
        print("Time for writeData: %e seconds" % (time.time() - start_time))
        
    def nextStep(self):
        """
        next time step when convergence is achieved

        Returns
        -------
        None.

        """
        self.system.nextStep()
        for gp in self.GPs:
            gp.nextStep()
    
    def resetToPreviousStep(self):
        """
        reset to previous solution when convergence is not achieved

        Returns
        -------
        None.

        """
        self.system.resetToPrevousStep()
        for gp in self.GPs:
            gp.resetToPreviousStep()
  
    def endScheme(self):
        for dataSave in self.dataArchiving:
            if dataSave.filePt is not None:
                dataSave.closeFile()
                dataSave.filePt=None        
                
    def solve(self, nsteps, startTime=0., endTime=1, tol=1e-6, absTol=1e-10, maxFailedSteps=100, maxNRIterations =20, lineSearch=False, beta=0.1):
        """
        nonlinear system solve

        Parameters
        ----------
        nsteps : int
            Number of time step.
        startTime : float, optional
            start time of simulation. The default is 0.
        endTime : float, optional
            end time of simulation. The default is 1.
        tol : float, optional
            Relative tolerance. The default is 1e-6.
        absTol : float, optional
            Absolute tolerance. The default is 1e-10.
        maxFailedSteps : int, optional
            Number of allowale fails. The default is 100.
        maxNRIterations : int, optional
            Number of allowale iterations in each step. The default is 20.
        Returns
        -------
        None.

        """
        systemSize=len(self.dofDict)
        fullSize = len(self.dofDict) + len(self.fixedDofs)
        
        if lineSearch:
            #self.system = System.lineSearchSystem(systemSize,fullSize,beta=beta) 
            self.system = System.nonlinearConjugateGradientSystem(systemSize,fullSize)
        else:
            self.system = System.nonLinearSystem(systemSize,fullSize)
        timeStep = (endTime-startTime)/nsteps # time step
        previousTime=startTime
        curTime=startTime
        failedSteps = 0
        
        istep =0
        while (curTime < endTime- absTol):
            start_time = time.time()
            curTime= previousTime + timeStep
            istep = istep+1
            if curTime > endTime:
                curTime = endTime
                timeStep = curTime-previousTime
            
            print(f"TIME STEP {istep}: curTime = {curTime:.6f}, previousTime ={previousTime:.6f}, timeStep={timeStep:.6f}")   
         
            # initialise norm
            norm0 = None
            # NR iteration
            iterationIndex=0 # counter
            convergence = False
            while True: 
                iterationIndex += 1
                if (iterationIndex > maxNRIterations):
                    print("allowable number of iterations is reached")
                    break
              
                # compute IPstate 
                self.computeIPVariable(curTime, timeStep, True)
                # compute right hand side
                self.computeRightHandSide(curTime)
                # check convergence by norm inf and norm 0 of right hand side
                normInf=self.system.normInfRHS()
                if (norm0 is None):
                    norm0 = self.system.normInfFint()+self.system.normInfFext()
                    
                defoEnerg = self.computeTotalDeformationEnergy()
                print(f"TIME STEP {iterationIndex}: defo Energy = {defoEnerg}")
                
                print(f"Iter {iterationIndex}: relative residual= {normInf/norm0:.6e}, absolue residual={normInf:.6e}")
                if np.isnan(normInf) or np.isnan(norm0):
                    break
                elif (norm0 < absTol):
                    print("convergence by absolute norm")
                    convergence = True
                    break
                elif normInf < tol*norm0:
                    print("convergence by relative norm")
                    convergence = True
                    break
                else:
                    # if not convergence, solve a linear system
                    # assemble stiffness matrix
                    if self.system.withStiffnessMatrix():
                        self.computeStiffness()
                    # solve linear system
                    ok = self.system.systemSolve()
                    if not(ok):
                        print("linear system is not successfully solved")
                        break
                    else:
                        self.system.updateSolution(True) # accept solution
            
            # reduced time step
            if not(convergence):
                istep -= 1
                timeStep *= 0.5
                # reset to previous data
                self.resetToPreviousStep()
                failedSteps += 1
                if failedSteps > maxFailedSteps:
                    print("simulation stops because of reaching maximal failed steps")
                    break
            else:
                # save data
                print("data archiving")
                self.writeData(curTime,istep)
                self.achiving(curTime)    
                self.nextStep()
                print("solving time in step %d: %e seconds" % (istep,time.time() - start_time))   
                print("--------------------DONE------------------------")
                previousTime=curTime
                
        self.endScheme()
  

    def qubo_solve(self, qubo , nsteps, 
                       maxNRIterations=100,
                       nBitsGradient=5,
                       etaMin = 0.,
                       etaMax = 1e-2,
                       nBitsRandom = 10,
                       alpha = 1e-2,
                       maxNumIncreased=5,
                       increasedFactor=1.5,
                       maxNumReduced=2,
                       reducedFactor=0.5,
                        startTime=0., endTime=1, 
                       tol=1e-6, absTol=1e-10, maxFailedSteps=100):
        systemSize=len(self.dofDict)
        fullSize = len(self.dofDict) + len(self.fixedDofs)
        #self.system = System.gradientDescentSystem(self, qubo, systemSize, fullSize, 
        #             nBitsGradient, etaMin, etaMax, 
        #             nBitsRandom, alpha)
        self.system = System.binarySearchSystem(self, qubo, systemSize, fullSize, 
                     nBitsGradient, etaMin, etaMax, 
                     nBitsRandom, alpha)
        
        timeStep = (endTime-startTime)/nsteps # time step
        previousTime=startTime
        curTime=startTime
        failedSteps = 0
        istep =0
        while (curTime < endTime- absTol):
            start_time = time.time()
            curTime= previousTime + timeStep
            istep = istep+1
            if curTime > endTime:
                curTime = endTime
                timeStep = curTime-previousTime
            print(f"TIME STEP {istep}: curTime = {curTime:.6f}, previousTime ={previousTime:.6f}, timeStep={timeStep:.6f}")   
            
            # initialise
            self.computeIPVariable(curTime, timeStep, True)
            defoEnerg = self.computeTotalDeformationEnergy()
            defoEnergInit = defoEnerg
            # compute right hand side
            self.computeRightHandSide(curTime)
            # initialise norm
            norm0 =  self.system.normInfFint()+self.system.normInfFext()
            # check convergence by norm inf and norm 0 of right hand side
            normInf=self.system.normInfRHS()
            #
            convergence = False
            failed = False
            print(f"Iter 0: relative residual= {normInf/norm0:.6e}, absolue residual={normInf:.6e}")
            if np.isnan(normInf) or np.isnan(norm0):
                failed = True
            elif (norm0 < absTol):
                print("convergence by absolute norm")
                convergence = True
            elif normInf < tol*norm0:
                print("convergence by relative norm")
                convergence = True
            # NR iteration
            iterationIndex=0 # counter
            numIncreased = 0
            numReduced = 0
            while not(convergence) and not(failed): 
                iterationIndex += 1
                if (iterationIndex > maxNRIterations):
                    print("allowable number of iterations is reached")
                    break
                # solve linear system
                ok = self.system.systemSolve()
                if not(ok):
                    failed = True
                    print("linear system is not successfully solved")
                    break
                else:
                    self.system.updateSolution(True) # accept solution
                
                self.computeIPVariable(curTime, timeStep, True)
                defoEnergCur = self.computeTotalDeformationEnergy()
                
                print(f"TIME STEP {iterationIndex}: defo Energy= {defoEnergCur}, previous smallest defo = {defoEnerg}")
                
                if defoEnergCur < defoEnerg:
                    defoEnergPrev = defoEnerg
                    defoEnerg=defoEnergCur
                    print(f"TIME STEP {iterationIndex}: defo Energy= {defoEnergCur}, previous smallest defo = {defoEnerg} diff = {(defoEnerg - defoEnergPrev)/defoEnergInit:.6e}")
                    # compute right hand side
                    self.computeRightHandSide(curTime)
                    normInf=self.system.normInfRHS()
                    print(f"Iter {iterationIndex}: relative residual= {normInf/norm0:.6e}, absolue residual={normInf:.6e}")
                    numIncreased += 1
                    if numIncreased > maxNumIncreased and numReduced>0:
                        numIncreased = 0
                        numReduced = 0
                        self.system.updateParameters(increasedFactor)
                    if np.isnan(normInf) or np.isnan(norm0):
                        failed=True
                        break
                    elif (norm0 < absTol):
                        print("convergence by absolute norm")
                        convergence = True
                        break
                    elif normInf < tol*norm0:
                        print("convergence by relative norm")
                        convergence = True
                        break
                else:
                    # back to brevious
                    self.system.updateSolution(False)
                    numReduced += 1
                    if numReduced > maxNumReduced:
                        numReduced = 0
                        numIncreased = 0
                        print("reducing interval")
                        self.system.updateParameters(reducedFactor)
                        
            # reduced time step
            if not(convergence) or failed:
                istep -= 1
                timeStep *= 0.5
                # reset to previous data
                self.resetToPreviousStep()
                failedSteps += 1
                if failedSteps > maxFailedSteps:
                    print("simulation stops because of reaching maximal failed steps")
                    break
            else:
                # save data
                print("data archiving")
                self.writeData(curTime,istep)
                self.achiving(curTime)    
                self.nextStep()
                print("solving time in step %d: %e seconds" % (istep,time.time() - start_time))   
                print("--------------------DONE------------------------")
                previousTime=curTime
                
        self.endScheme()
            
