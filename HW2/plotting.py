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
    solutionMatrix = PDEobj.solutionMatrix[indexBeforeMinT:indexAfterMaxT,:]
    return Tarray, Xarray, solutionMatrix

def makeContourPlot(fig, ax, Tarray, Xarray, matrix, colormap):
    X, Y = np.meshgrid(Xarray, Tarray)
    mylevels = np.linspace(np.min(matrix), np.max(matrix), 50)

    contour_plot = ax.contourf(X, Y, matrix, levels=mylevels, cmap=colormap)
    min_tick = np.min(matrix)
    max_tick = np.max(matrix)
    min_tick_label = f'{min_tick:.2f}'
    max_tick_label = f'{max_tick:.2f}'
    if min_tick < 0:
        my_ticks = [min_tick, 0, max_tick]
        my_tick_labels = [min_tick_label, f'{0:.2f}', max_tick_label]
    else:
        my_ticks = [min_tick, max_tick]
        my_tick_labels = [min_tick_label, max_tick_label]
    
    cbar = fig.colorbar(contour_plot)
    cbar.set_ticks(ticks=my_ticks, labels=my_tick_labels)

    return ax

def plotEverything(PDEobj, minT, maxT):
    fig, ax = plt.subplots(1, 3, figsize=(7,2), dpi=160, layout='constrained')

    Tarray, Xarray, solutionMatrix = getPlottingArrays(PDEobj, minT, maxT)
    trueSolution = aux_funcs.calculateTrueSolution(Tarray, Xarray)

    ax[0].plot(Xarray, solutionMatrix[-1,:], color='black', linewidth=1, label='Num.')
    ax[0].plot(Xarray, trueSolution[-1,:], color='black', linestyle=':', linewidth=1, label='True')
    ax[1] = makeContourPlot(fig, ax[1], Tarray, Xarray, solutionMatrix, colormaps['twilight'])
    ax[2] = makeContourPlot(fig, ax[2], Tarray, Xarray, abs(solutionMatrix - trueSolution), colormaps['inferno'])

    ax[0].set_xlabel('x')
    ax[1].set_xlabel('x')
    ax[2].set_xlabel('x')

    ax[0].set_ylabel('u(x,T)')
    ax[1].set_ylabel('Time')
    ax[2].set_yticks([])

    method = PDEobj.method
    dxString = str(PDEobj.pars['dx'])
    dtString = str(PDEobj.pars['dt'])
    plt.suptitle('Numerical solution of ' + method + ' scheme; Δx = ' + dxString + ', Δt = ' + dtString)

    ax[0].set_title('Solutions at T=10', fontsize=10)
    ax[1].set_title('Numerical Solution', fontsize=10)
    ax[2].set_title('Absolute Error', fontsize=10)

    saveStr = method + '_dx=' + dxString + '_dt=' + dtString + '.png'
    plt.savefig('Figures/' + saveStr)

def animateNumericalSolution(PDEobj, minT, maxT, save):
    import matplotlib.animation as animation

    Tarray, Xarray, solutionMatrix = getPlottingArrays(PDEobj, minT, maxT)

    fig, ax = plt.subplots(1, 1, figsize=(6,2), dpi=160)

    line1 = ax.plot(Xarray, aux_funcs.initialCondition(Xarray), label='t = 0', color='black')[0]
    leg = ax.legend(loc='upper right')

    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')

    method = PDEobj.method
    dxString = str(PDEobj.pars['dx'])
    dtString = str(PDEobj.pars['dt'])
    ax.set_title('Num. sol. of ' + method + ' scheme; Δx = ' + dxString + ', Δt = ' + dtString) 

    # Controls how fast the animation plays
    skip = int(maxT - minT) # In this case, skip = 10, so we skip every 10 timesteps

    def update(frame):
        # Update data in the plot
        line1.set_xdata(Xarray)
        line1.set_ydata(solutionMatrix[frame*skip,:])
        # Update legend to show current time
        leg.get_texts()[0].set_text(f"t = {Tarray[frame*skip]:.2f}")
        return (line1)

    anim = animation.FuncAnimation(fig=fig, func=update, frames=int(len(Tarray) / skip), interval=10, blit=True)
    plt.show()

    if (save == True):
        writer = animation.PillowWriter(fps=30)
        saveStr = 'Animation_' + method + '_dx=' + dxString + '_dt=' + dtString + '.gif'
        anim.save('Solution_GIFs/' + saveStr, writer=writer)

    return anim