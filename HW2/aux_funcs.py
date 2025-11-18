import numpy as np

'''
If input is a a float, returns the initial condition at that point
If input is an array, returns an array of the initial condition at each point in the array
'''

def initialCondition(x):
    if isinstance(x, float):
        if 0 <= x and x < 1/3:
            return np.cos(6*np.pi*x)
        elif 1/3 <= x and x < 2/3:
            return 0
        elif 2/3 <= x and x <= 1:
            return np.cos(12*np.pi*x)
        else:
            return -1
    else:
        retArray = np.zeros((len(x)))
        for i in range(len(x)):
            if 0 <= x[i] and x[i] < 1/3:
                retArray[i] = np.cos(6*np.pi*x[i])
            elif 1/3 <= x[i] and x[i] < 2/3:
                retArray[i] = 0
            elif 2/3 <= x[i] and x[i] <= 1:
                retArray[i] = np.cos(12*np.pi*x[i])
            else:
                retArray[i] = -1

    return retArray

'''
Returns the true solution at a point (x,t) using u(x,t) = u(x-t,0)
'''

def trueSolution(t,x):
    return initialCondition(x-t - int(x-t) + 1 if x-t <= 0 else x-t - int(x-t))

'''
Given an array of times and x values, returns a matrix of the true solution at all points.
'''

def calculateTrueSolution(Tarray, Xarray):
    rows, cols = np.meshgrid(Tarray, Xarray)
    trueSolutionVectorized = np.vectorize(trueSolution)
    return np.transpose(trueSolutionVectorized(rows, cols))

'''
interpolateMatrix(A, Tarray, Xarray, t, x)

Accepts:    - A matrix, A, of dimensions len(Tarray) x len(Xarray)
            - Arrays of time values, Tarray, and x-values, Xarray
            - Floats t and x within the endpoints of Tarray and Xarray
Returns:    - The value of A at (t,x) by interpolating the four neighbors of (t,x).
'''

def interpolateMatrix(A, Tarray, Xarray, t, x):
    xIndex = np.searchsorted(Xarray, x, 'left') - 1
    tIndex = np.searchsorted(Tarray, t, 'left') - 1

    if not x in Xarray:
        # Need to interpolate x
        if not t in Tarray:
            # Need to interpolate x and t
            xLeft = Xarray[xIndex]
            xRight = Xarray[xIndex + 1]
            tPrior = Tarray[tIndex]
            tAfter = Tarray[tIndex + 1]
            fQ11 = A[tIndex, xIndex]
            fQ12 = A[tIndex, xIndex + 1]
            fQ21 = A[tIndex + 1, xIndex]
            fQ22 = A[tIndex + 1, xIndex + 1]

            xVec = np.array([tAfter - t, t - tPrior])
            fMat = np.array([[fQ11, fQ12], [fQ21, fQ22]])
            yVec = np.array([[xRight - x], [x - xLeft]])
            interpolatedValue = (1 / ((xRight - xLeft)*(tAfter - tPrior))) * np.matmul(np.matmul(xVec, fMat), yVec)
        else:
            # Only need to interpolate x
            xLeft = Xarray[xIndex]
            xRight = Xarray[xIndex + 1]
            fLeft = A[tIndex, xIndex]
            fRight = A[tIndex, xIndex + 1]
            interpolatedValue = (xRight - x)/(xRight - xLeft) * fLeft + (x - xLeft)/(xRight - xLeft) * fRight
    elif not t in Tarray:
        # Only need to interpolate t
        tPrior = Tarray[tIndex]
        tAfter = Tarray[tIndex + 1]
        fPrior = A[tIndex, xIndex]
        fAfter = A[tIndex + 1, xIndex]
        interpolatedValue = (tAfter - t)/(tAfter - tPrior) * fPrior + (t - tPrior)/(tAfter - tPrior) * fAfter
    else:
        interpolatedValue = A[tIndex, xIndex]

    return interpolatedValue

'''
interpolateMatrixVectorized

A vectorized verson of interpolateMatrix that can accept t and x as arrays and evaluate the
matrix A over another matrix defined by t and x.

This is used for plotting. Interpolating over a PDE solution over coarse arrays t and x is faster
than plotting the whole solution over the entirety of Tarray and Xarray
'''

def interpolateMatrixVectorized(A, Tarray, Xarray, t, x):
    T, X = np.meshgrid(t, x)
    interpolateMatrixVec = np.vectorize(interpolateMatrix, excluded={0,1,2})
    return np.transpose(interpolateMatrixVec(A, Tarray, Xarray, T, X))