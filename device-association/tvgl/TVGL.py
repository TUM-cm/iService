import numpy as np
import numpy.linalg as alg

def TVGL(data, lengthOfSlice, lamb, beta, indexOfPenalty,
         verbose = False, eps = 3e-3, epsAbs = 1e-3, epsRel = 1e-3):        
    if indexOfPenalty == 1:
        print('Use l-1 penalty function')
        from inferGraphL1 import TGraphVX
        from inferGraphL1 import semidefinite
        from inferGraphL1 import log_det
        from inferGraphL1 import trace
        from inferGraphL1 import norm
    elif indexOfPenalty == 2:
        print('Use l-2 penalty function')
        from inferGraphL2 import TGraphVX
        from inferGraphL2 import semidefinite
        from inferGraphL2 import log_det
        from inferGraphL2 import trace
        from inferGraphL2 import norm
    elif indexOfPenalty == 3:
        print('Use laplacian penalty function')
        from inferGraphLaplacian import TGraphVX
        from inferGraphLaplacian import semidefinite
        from inferGraphLaplacian import log_det
        from inferGraphLaplacian import trace
        from inferGraphLaplacian import norm
    elif indexOfPenalty == 4:
        print('Use l-inf penalty function')
        from inferGraphLinf import TGraphVX
        from inferGraphLinf import semidefinite
        from inferGraphLinf import log_det
        from inferGraphLinf import trace
        from inferGraphLinf import norm
    else:
        print('Use perturbation node penalty function')
        from inferGraphPN import TGraphVX
        from inferGraphPN import semidefinite
        from inferGraphPN import log_det
        from inferGraphPN import trace
        from inferGraphPN import norm
    
    numberOfTotalSamples = data.shape[0]
    timestamps = int(numberOfTotalSamples/lengthOfSlice)    
    size = data.shape[1]
    # Generate empirical covariance matrices
    empCovSet = []    # list of array
    sampleSet = []    # list of array
    k = 0
    for i in range(timestamps):
        # Generate the slice of samples for each timestamp from data
        k_next = min(k + lengthOfSlice, numberOfTotalSamples)
        samples = data[k : k_next, :]
        k = k_next
        sampleSet.append(samples)
        empCov = GenEmpCov(sampleSet[i].T)
        empCovSet.append(empCov)
    
    print(sampleSet.__len__())
    print('lambda = %s, beta = %s'%(lamb, beta))
    
    # Define a graph representation to solve
    gvx = TGraphVX()   
    for i in range(timestamps):
        n_id = i
        S = semidefinite(size, name='S')
        obj = -log_det(S) + trace(empCovSet[i] * S) #+ alpha*norm(S,1)
        gvx.AddNode(n_id, obj)
        if i > 0: # Add edge to previous timestamp
            prev_Nid = n_id - 1
            currVar = gvx.GetNodeVariables(n_id)
            prevVar = gvx.GetNodeVariables(prev_Nid)            
            edge_obj = beta * norm(currVar['S'] - prevVar['S'], indexOfPenalty) 
            gvx.AddEdge(n_id, prev_Nid, Objective = edge_obj)
        #Add rake nodes, edges
        gvx.AddNode(n_id + timestamps)
        gvx.AddEdge(n_id, n_id + timestamps, Objective= lamb * norm(S,1))
    
    # need to write the parameters of ADMM
    gvx.Solve(EpsAbs=epsAbs, EpsRel=epsRel, Verbose = verbose)
    #gvx.Solve(MaxIters=2, EpsAbs=epsAbs, EpsRel=epsRel, Verbose = verbose)
    #gvx.Solve(MaxIters = 700, Verbose = True, EpsAbs=eps_abs, EpsRel=eps_rel)
    # gvx.Solve( NumProcessors = 1, MaxIters = 3)
    
    # Extract the set of estimated theta 
    thetaSet = []
    thetaEst_previous = None
    for nodeID in range(timestamps):
        val = gvx.GetNodeValue(nodeID, 'S')
        thetaEst = upper2FullTVGL(val, eps)
        if verbose:
            print('\nAt node = ', nodeID, '-----------------')
            print('EmpCov = \n', empCovSet[nodeID])
            print('Cov = \n', alg.inv(thetaEst))
            print('Theta_est=\n', thetaEst)
            if nodeID != 0:
                print('Theta_diff = \n', thetaEst - thetaEst_previous)
                print('FRONORM = ', alg.norm(thetaEst - thetaEst_previous, 'fro'))
            thetaEst_previous = thetaEst
        thetaSet.append(thetaEst)
    return thetaSet

def GenEmpCov(samples, useKnownMean = False, m = 0):
    # samples should be array
    _, samplesPerStep = samples.shape
    if useKnownMean == False:
        m = np.mean(samples, axis = 1)
    empCov = 0
    for i in range(samplesPerStep):
        sample = samples[:,i]
        empCov = empCov + np.outer(sample - m, sample -m)
    empCov = empCov/samplesPerStep
    return empCov
    
def upper2FullTVGL(a, eps = 0):
    # a should be array
    ind = (a < eps) & (a > -eps)
    a[ind] = 0
    n = int((-1  + np.sqrt(1+ 8*a.shape[0]))/2)  
    A = np.zeros([n,n])
    A[np.triu_indices(n)] = a 
    d = A.diagonal()
    A = np.asarray((A + A.T) - np.diag(d))             
    return A   
