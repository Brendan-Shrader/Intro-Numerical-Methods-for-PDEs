import numpy as np

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

def trueSolution(t,x):
    return initialCondition(x-t - int(x-t) + 1 if x-t <= 0 else x-t - int(x-t))

def calculateTrueSolution(TarrayPlotting, Xarray):
    rows, cols = np.meshgrid(TarrayPlotting, Xarray)
    trueSolutionVectorized = np.vectorize(trueSolution)
    return np.transpose(trueSolutionVectorized(rows, cols))

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

def interpolateMatrixVectorized(A, Tarray, Xarray, t, x):
    T, X = np.meshgrid(t, x)
    interpolateMatrixVec = np.vectorize(interpolateMatrix, excluded={0,1,2})
    return np.transpose(interpolateMatrixVec(A, Tarray, Xarray, T, X))