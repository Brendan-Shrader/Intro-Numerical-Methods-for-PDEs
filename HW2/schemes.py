'''
methods.py
'''

import numpy as np

def getParameters(PDEobj):
    parsList = ['dt', 'dx', 'T', 'J', 'Xarray']
    return [PDEobj.pars[parameter] for parameter in parsList]

def initializeSolutionMatrix(T, J, Xarray, initialCondition):
    solutionMatrix = np.zeros((T+1, J+1))
    solutionMatrix[0,:] = np.array([initialCondition(x) for x in Xarray])
    return solutionMatrix

# Gets parameters and sets up initial solution for all schemes
def initializeScheme(PDEobj):
    dt, dx, T, J, Xarray = getParameters(PDEobj)
    solutionMatrix = initializeSolutionMatrix(T, J, Xarray, PDEobj.initialCondition)
    return dt, dx, T, J, Xarray, solutionMatrix

'''
Upwind Scheme
'''

def upwindStep(Uprev, r):
    Unext = (1-r)*Uprev + r*np.roll(Uprev, 1)
    Unext[0] = Uprev[-2]
    return Unext

def solveUpwind(PDEobj):
    dt, dx, T, J, Xarray, U = initializeScheme(PDEobj)
    r = dt / dx
    for n in range(0, T):
        U[n+1,:] = upwindStep(U[n,:], r)

    return U

'''
Lax-Friedrichs Scheme
'''

def solveLaxFriedrichs(PDEobj):
    dt, dx, T, J, Xarray, U = initializeScheme(PDEobj)
    r = dt / dx
    for n in range(0, T):
        U[n+1,0] = (1 - r)*U[n,1]/2 + (1 + r)*U[n,J-1]/2
        U[n+1,J] = (1 - r)*U[n,1]/2 + (1 + r)*U[n,J-1]/2
        for j in range(1, J):
            U[n+1,j] = (1 - r)*U[n,j+1]/2 + (1 + r)*U[n,j-1]/2

    return U

'''
CIR Scheme
'''

def CIRsetupXpredicted(dt, Xarray, a):
    xShifted = Xarray - a*dt
    decimalParts, _ = np.modf(xShifted)
    return np.abs(np.where(decimalParts > 0, decimalParts, decimalParts + 1))

def CIRinterpolateOptimized(Uprev, Xarray, x_index_left, x):
    xIndexRight = x_index_left + 1
    xLeft = Xarray[x_index_left]
    xRight = Xarray[xIndexRight]
    Uleft = Uprev[x_index_left]
    Uright = Uprev[xIndexRight]
    return Uleft + (Uright - Uleft)/(xRight - xLeft) * (x - xLeft)

def CIRstepOptimized(Uprev, x_predicted_to_left_index, x_predicted, interpolationFunction):
    return interpolationFunction(Uprev, x_predicted_to_left_index, x_predicted)

def solveCIRoptimized(PDEobj):
    dt, dx, T, J, Xarray, solutionMatrix = initializeScheme(PDEobj)

    # Calculates x - aΔt for each x in Xarray, shifted to be in [0,1]
    a = 1
    x_predicted = CIRsetupXpredicted(dt, Xarray, a)
    # Precalculate where x - aΔt lands
    x_predicted_to_left_index = np.array([np.searchsorted(Xarray, x, 'left') - 1 for x in x_predicted])

    # Vectorize CIRinterpolate to make each step easier
    interpolationFunction = np.vectorize(lambda Uprev, x_index_left, x: CIRinterpolateOptimized(Uprev, Xarray, x_index_left, x), excluded={0, 1, 2})

    # Solve the system
    for n in range(1, T+1):
        solutionMatrix[n,:] = CIRstepOptimized(solutionMatrix[n-1,:], x_predicted_to_left_index, x_predicted, interpolationFunction)

    return solutionMatrix

'''
CIR Scheme with BFECC
'''

def solveCIRBFECCoptimized(PDEobj):
    dt, dx, T, J, Xarray, solutionMatrix = initializeScheme(PDEobj)

    # Calculates x - aΔt for each x in Xarray, shifted to be in [0,1]
    a = 1
    x_predicted = CIRsetupXpredicted(dt, Xarray, a)
    x_predicted_reverse_velocity = CIRsetupXpredicted(dt, Xarray, -a)
    # Precalculate where x - aΔt lands
    x_predicted_to_left_index = np.array([np.searchsorted(Xarray, x, 'left') - 1 for x in x_predicted])
    x_predicted_reverse_to_left_index = np.array([np.searchsorted(Xarray, x, 'left') - 1 for x in x_predicted_reverse_velocity])

    # Vectorize CIRinterpolate to make each step easier
    interpolationFunction = np.vectorize(lambda Uprev, x_index_left, x: CIRinterpolateOptimized(Uprev, Xarray, x_index_left, x), excluded={0, 1, 2})

    # Solve the system
    for n in range(1, T+1):
        Uprev = solutionMatrix[n-1,:]
        UdotNext = CIRstepOptimized(solutionMatrix[n-1,:], x_predicted_to_left_index, x_predicted, interpolationFunction)
        UdotPrev = CIRstepOptimized(UdotNext, x_predicted_reverse_to_left_index, x_predicted_reverse_velocity, interpolationFunction)
        solutionMatrix[n,:] = CIRstepOptimized(Uprev + (Uprev - UdotPrev)/2, x_predicted_to_left_index, x_predicted, interpolationFunction)

    return solutionMatrix

'''
Lax-Wendroff Scheme
'''

def solveLaxWendroff(PDEobj):
    dt, dx, T, J, Xarray, U = initializeScheme(PDEobj)
    r = dt / dx
    for n in range(0, T):
        U[n+1, 0] = (r**2 - r)*U[n,1]/2 + (1 - r**2)*U[n,0] + (r**2 + r)*U[n,J-1]/2
        U[n+1, J] = (r**2 - r)*U[n,1]/2 + (1 - r**2)*U[n,J] + (r**2 + r)*U[n,J-1]/2
        for j in range(1, J):
            U[n+1, j] = (r**2 - r)*U[n,j+1]/2 + (1 - r**2)*U[n,j] + (r**2 + r)*U[n,j-1]/2

    return U

'''
MacCormack Scheme
'''

def downwindStep(Uprev, r):
    Unext = (1+r)*Uprev - r*np.roll(Uprev, -1)
    Unext[-1] = Uprev[1]
    return Unext

def solveMacCormack(PDEobj):
    dt, dx, T, J, Xarray, U = initializeScheme(PDEobj)
    r = dt / dx
    for n in range(0, T):
        Ustar = upwindStep(U[n,:], r)
        Ustar = downwindStep(Ustar, r)
        U[n+1,:] = (U[n,:] + Ustar)/2

    return U

'''
Implicit Lax-Wendroff Scheme
'''

def solveImplicitLaxWendroff(PDEobj):
    #from aux_funcs import seidel
    dt, dx, T, J, Xarray, U = initializeScheme(PDEobj)
    r = dt / dx
    A = np.zeros((J+1, J+1))

    """ for j in range(1,J):
        A[j,j+1] = (r**2 + r)/2
        A[j,j] = 1 - r**2
        A[j,j-1] = (r**2 - r)/2
    
    A[0,1] = (r**2 + r)/2
    A[0,0] = 1 - r**2
    A[0,J-1] = (r**2 - r)/2
    A[J,1] = (r**2 + r)/2
    A[J,J] = 1 - r**2
    A[0,J-1] = (r**2 - r)/2 """

    for j in range(1,J):
        A[j,j+1] = (r**2 - r)/2
        A[j,j] = 1 - r**2
        A[j,j-1] = (r**2 + r)/2
    
    A[0,1] = (r**2 - r)/2
    A[0,0] = 1 - r**2
    A[0,J-1] = (r**2 + r)/2
    A[J,1] = (r**2 - r)/2
    A[J,J] = 1 - r**2
    A[0,J-1] = (r**2 + r)/2

    Ainv = np.linalg.inv(A)

    for n in range(T):
        U[n+1,:] = np.matmul(Ainv, U[n,:])
    
    return U