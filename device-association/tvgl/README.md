https://github.com/davidhallac/TVGL

## error 1:

Traceback (most recent call last):
  File "financeInference.py", line 414, in <module>
    gvx, empCov_set = solveProblem(gvx, index_penalty, cov_mode, alpha, beta, timesteps, timeShift, Cov_set, use_kernel, sigma, sample_set, empCov_set, eps_abs, eps_rel)
  File "financeInference.py", line 235, in solveProblem
    gvx.Solve(EpsAbs=eps_abs, EpsRel=eps_rel,Verbose=True,MaxIters=500)
  File "C:\Daten\Downloads\TVGL\tvgl\PaperCode\inferGraphPN.py", line 147, in Solve
    Verbose)
  File "C:\Daten\Downloads\TVGL\tvgl\PaperCode\inferGraphPN.py", line 437, in __SolveADMM
    map(ADMM_x, node_list)
  File "C:\Daten\Downloads\TVGL\tvgl\PaperCode\inferGraphPN.py", line 1124, in ADMM_x
    a = numpy.zeros(mat_shape)
TypeError: 'float' object cannot be interpreted as an index

numpy.zeros(mat_shape) --> mat_shape should be int type
mat_shape = (int(numpymat.shape[1] * ( numpymat.shape[1]+1 )/2.0),)

## error 2:

Use laplacian penalty function
3
lambda = 2.5, beta = 12
Distributed ADMM (4 processors)
Iteration 1
Traceback (most recent call last):
File "D:\workplace\eclipse_workplace\TVGL\exampleTVGL.py", line 20, in 
thetaSet = tvgl.TVGL(data, lengthOfSlice, lamb, beta, indexOfPenalty = 3, verbose=True)
File "D:\workplace\eclipse_workplace\TVGL\TVGL.py", line 66, in TVGL
gvx.Solve(EpsAbs=epsAbs, EpsRel=epsRel, Verbose = verbose)
File "D:\workplace\eclipse_workplace\TVGL\inferGraphLaplacian.py", line 147, in Solve
Verbose)
File "D:\workplace\eclipse_workplace\TVGL\inferGraphLaplacian.py", line 437, in __SolveADMM
pool.map(ADMM_x, node_list)
File "D:\software\Anaconda2\Lib\multiprocessing\pool.py", line 251, in map
return self.map_async(func, iterable, chunksize).get()
File "D:\software\Anaconda2\Lib\multiprocessing\pool.py", line 567, in get
raise self._value
TypeError: 'NoneType' object has no attribute 'getitem'

adapt

inferGraphL1.py
inferGraphL2.py
inferGraphLaplacian.py
inferGraphLinf.py
inferGraphPN.py

replace

        pool.map(ADMM_x, node_list)
        pool.map(ADMM_z, edge_list)
        pool.map(ADMM_u, edge_list)

with the following:

        map(ADMM_x, node_list)
        map(ADMM_z, edge_list)
        map(ADMM_u, edge_list)
