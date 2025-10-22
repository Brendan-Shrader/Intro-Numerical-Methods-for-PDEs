import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

import aux_funcs

'''
getPlottingArrays(PDEobj, minT, maxT)

Accepts: A solved PDEclass object, and minimum/maximum plotting times
'''
def getPlottingArrays(PDEobj, minT, maxT):
    Xarray = PDEobj.pars['Xarray']
    Tarray = PDEobj.pars['Tarray']
    indexBeforeMinT = np.maximum(np.searchsorted(Tarray, minT, 'left') - 1, 0)
    indexAfterMaxT = np.minimum(np.searchsorted(Tarray, maxT, 'right'), PDEobj.pars['T'] + 1)
    Tarray = Tarray[indexBeforeMinT:indexAfterMaxT]
    solutionMatrixPlotting = PDEobj.solutionMatrix[indexBeforeMinT:indexAfterMaxT,:]
    return Tarray, Xarray, solutionMatrixPlotting

def makeContourPlot(Tarray, Xarray, matrix, colormap):
    fig, ax = plt.subplots(1, 1, figsize=(6,2), dpi=160)
    X, Y = np.meshgrid(Xarray, Tarray)
    mylevels = np.linspace(np.min(matrix), np.max(matrix), 50)

    contour_plot = ax.contourf(X, Y, matrix, levels=mylevels, cmap=colormap)
    fig.colorbar(contour_plot)

    return fig, ax

def plotAbsoluteError(PDEobj, minT, maxT):
    # Get Xarray, Tarray, and solutionMatrix between minT and maxT
    Tarray, Xarray, solutionMatrix = getPlottingArrays(PDEobj, minT, maxT)

    # Calculate the true solution at the same values
    trueSolution = aux_funcs.calculateTrueSolution(Tarray, Xarray)

    fig, ax = makeContourPlot(Tarray, Xarray, abs(solutionMatrix - trueSolution), colormaps['inferno'])

    method = PDEobj.method
    dxString = str(PDEobj.pars['dx'])
    dtString = str(PDEobj.pars['dt'])
    ax.set_title('Abs. error for ' + method + ' scheme; Δx = ' + dxString + ', Δt = ' + dtString)

def plotNumericalSolution(PDEobj, minT, maxT):
    Tarray, Xarray, solutionMatrix = getPlottingArrays(PDEobj, minT, maxT)

    fig, ax = makeContourPlot(Tarray, Xarray, solutionMatrix, colormaps['twilight'])

    method = PDEobj.method
    dxString = str(PDEobj.pars['dx'])
    dtString = str(PDEobj.pars['dt'])
    ax.set_title('Num. sol. of ' + method + ' scheme; Δx = ' + dxString + ', Δt = ' + dtString)

def plotNumericalSolutionInterpolated(PDEobj, minT, maxT):
    XarrayCoarse = np.linspace(0, 1, 1000)
    TarrayCoarse = np.linspace(minT, maxT, 1000)

    solutionMatrixInterpolated = aux_funcs.interpolateMatrixVectorized(
        PDEobj.solutionMatrix, PDEobj.pars['Tarray'], PDEobj.pars['Xarray'], TarrayCoarse, XarrayCoarse)

    fig, ax = makeContourPlot(TarrayCoarse, XarrayCoarse, solutionMatrixInterpolated, colormaps['twilight'])

    method = PDEobj.method
    dxString = str(PDEobj.pars['dx'])
    dtString = str(PDEobj.pars['dt'])
    ax.set_title('Num. sol. of ' + method + ' scheme; Δx = ' + dxString + ', Δt = ' + dtString) 

def plotTrueSolution(PDEobj, minT, maxT):
    Tarray, Xarray, *_ = getPlottingArrays(PDEobj, minT, maxT)
    trueSolution = aux_funcs.calculateTrueSolution(Tarray, Xarray)
    fig, ax = makeContourPlot(Tarray, Xarray, trueSolution, colormaps['twilight'])
    ax.set_title('True Solution')

def plotTrueSolutionInterpolated(PDEobj, minT, maxT):
    XarrayCoarse = np.linspace(0, 1, 1000)
    TarrayCoarse = np.linspace(minT, maxT, 1000)
    trueSolutionInterpolated = aux_funcs.calculateTrueSolution(TarrayCoarse, XarrayCoarse)
    fig, ax = makeContourPlot(TarrayCoarse, XarrayCoarse, trueSolutionInterpolated, colormaps['twilight'])
    ax.set_title('True Solution')