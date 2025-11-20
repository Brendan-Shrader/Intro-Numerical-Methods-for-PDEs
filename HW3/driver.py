import numpy as np
import matplotlib.pyplot as plt
from functions import *

DT_SCALE = 0.3    # dt is allowed to be at most dx*DT_SCALE

plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10})

class ConservativePDE:
    def __init__(self, a, b, num_cells, f):
        '''Accepts: - Left and right bounds, <a> and <b>
                    - The number of cells to simulate, <num_cells>
                    - A function for the conservation PDE, <f>'''
        '''Returns: - Nothing'''

        self.num_cells = num_cells
        self.f = f
        self.a = a
        self.b = b
        self.J = num_cells - 1
        self.dx = (b - a) / num_cells
        self.cell_centers = np.arange(a, b, self.dx) + self.dx / 2 # Stores the centers of the cells
        self.cell_boundaries = np.arange(a, b + self.dx, self.dx)
    
    # Sets the initial condition of the PDE
    def setInitialCondition(self, initial_condition):
        '''Accepts: A function for the initial condition u(x,0), <initial_condition>'''
        '''Returns: Nothing'''

        self.initial_condition = initial_condition

    # Main function to solve the PDE
    def solveMUSCL(self, tmax, slope_limiter='minmod'):
        '''Accepts: - The maximum simulation time, <tmax>
                    - The slope limiter method, <slope_limiter> (default 'minmod')'''
        '''Returns: - The solution matrix'''

        # Check if initial condition has been set
        if not hasattr(self, 'initial_condition'):
            print("Set initial condition before solving.")
            return -1

        # Set the limiter input by the user
        self.slope_limiter_str = slope_limiter
        match slope_limiter:
            case 'minmod': self.slope_limiter = minmodLimiterVectorized
            case 'superbee': self.slope_limiter = superbeeLimiterVectorized
            case _:
                print('Invalid slope_limiter input to solveMUSCL(), using minmod...')
                self.slope_limiter = minmodLimiterVectorized
                self.slope_limiter_str = 'minmod'
        
        # Initialize solution matrix
        U = self.__initialize_solution__(tmax)

        # Improved Euler scheme
        for t in range(1, self.T+1):
            Ustar1 = self.__MUSCLstep__(U[t-1])
            Ustar2 = self.__MUSCLstep__(Ustar1)
            U[t] = (U[t-1] + Ustar2) / 2

        self.solution_matrix = U
        return U
    
    # Called by solveMUSCL() to performs one timestep of the scheme
    def __MUSCLstep__(self, Uprev):
        '''Accepts: The previous solution, u(x,t_{n-1}), <Uprev>'''
        '''Returns: The solution at the next timestep, u(x,t_{n})'''
        
        Phi = getLimitedSlopes(self, Uprev) # Polynomial Reconstruction

        F = getNumericalFluxes(self, Uprev, Phi) # Numerical Flux Calculations

        return Uprev - (self.dt / self.dx) * F

    # Called by solveMUSCL() to Initialize the solution matrix using the initial condition
    def __initialize_solution__(self, tmax):
        self.__set_time_attributes__(tmax)
        U = np.zeros((self.T + 1, self.J + 1))
        ic_vectorized = np.vectorize(self.initial_condition)
        U[0] = ic_vectorized(self.cell_centers)
        return U

    # Called by __init__() to set more object attributes
    def __set_time_attributes__(self, tmax):
        max_dt = self.dx * DT_SCALE
        self.tmax = tmax
        self.T = int(np.ceil(tmax / max_dt))
        self.dt = tmax / self.T
        self.tarray = np.linspace(0, tmax, self.T + 1)

    # Allows the user to get the numerical solution near a specified time
    def solutionAtTime(self, t):
        '''Accepts: - A time between 0 and tmax (inclusive)'''
        '''Returns: - The numerical solution near t'''
        if t < 0:
            print('Negative time input to solutionAtTime()')
            return -1 * np.ones(len(self.cell_centers))
        elif t > self.tmax:
            print('Time beyond tmax input to solutionAtTime()')
            return -1 * np.ones(len(self.cell_centers))
        
        t_index = np.searchsorted(self.tarray, t, side='left')
        if t_index == len(self.tarray): t_index -= 1
        return self.solution_matrix[t_index]



''' ___ PROBLEM 1 ___ '''

p1_f = lambda x: np.pow(x,2) / 2 # Burger's Equation

p1obj = ConservativePDE(0, 1, 100, p1_f) # Made PDE object

p1obj.setInitialCondition(lambda x: 1 + np.sin(2*np.pi*x)) # Set the initial condition

p1_sol_matrix = p1obj.solveMUSCL(tmax=1, slope_limiter='superbee') # Solve

if isinstance(p1_sol_matrix, int): exit()



# Plotting

X, Y = np.meshgrid(p1obj.cell_centers, p1obj.tarray)
fig, ax = plt.subplots(1, 1, figsize=(4.5,2.5), dpi=160)
cfplot = ax.contourf(X, Y, p1_sol_matrix, levels=100)
fig.colorbar(cfplot)
ax.set_ylabel('Time', rotation=0, labelpad=15)
ax.set_xlabel('x')
ax.set_title('Heatmap of u(x,t); ' + p1obj.slope_limiter_str + ' limiter')
fig.tight_layout()
plt.savefig('Figures/problem-1-heatmap-limiter=' + p1obj.slope_limiter_str + '.png', bbox_inches='tight')

fig, ax = plt.subplots(1, 1, figsize=(4.5,2.5), dpi=160)
ax.plot(p1obj.cell_centers, p1obj.solutionAtTime(0), label='Initial Condition', linestyle=':', color='k', linewidth=0.75)
ax.plot(p1obj.cell_centers, p1obj.solutionAtTime(1), label='Solution at T=1')
ax.set_ylabel('u(x,t)', rotation=0, labelpad=15)
ax.set_xlabel('x')
ax.set_title('u(x,t) solution; ' + p1obj.slope_limiter_str + ' limiter')
ax.legend(fontsize=9)
fig.tight_layout()
plt.savefig('Figures/problem-1-solution-limiter=' + p1obj.slope_limiter_str + '.png', bbox_inches='tight')



''' ___ PROBLEM 2 ___ '''

def p2_initial_condition(x):
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
    
p2_f = lambda x: x # Advection equation

p2obj = ConservativePDE(0, 1, 100, p2_f) # Make PDE object

p2obj.setInitialCondition(p2_initial_condition) # Set initial condition

p2_sol_matrix = p2obj.solveMUSCL(tmax=10, slope_limiter='superbee') # Solve

if isinstance(p2_sol_matrix, int): exit()



# Plotting

X, Y = np.meshgrid(p2obj.cell_centers, p2obj.tarray)
fig, ax = plt.subplots(1, 1, figsize=(4.5,2.5), dpi=160)
cfplot = ax.contourf(X, Y, p2_sol_matrix, levels=100)
fig.colorbar(cfplot)
ax.set_ylabel('Time', rotation=0, labelpad=10)
ax.set_xlabel('x')
ax.set_title('Heatmap of u(x,t); ' + p2obj.slope_limiter_str + ' limiter')
fig.tight_layout()
plt.savefig('Figures/problem-2-heatmap-limiter=' + p2obj.slope_limiter_str + '.png', bbox_inches='tight')

fig, ax = plt.subplots(1, 1, figsize=(4.5,2.5), dpi=160)
ax.plot(p2obj.cell_centers, p2obj.solutionAtTime(0), label='Initial Condition', linestyle=':', color='k', linewidth=0.75)
ax.plot(p2obj.cell_centers, p2obj.solutionAtTime(1), label='Solution at T=1')
ax.plot(p2obj.cell_centers, p2obj.solutionAtTime(5), label='Solution at T=5')
ax.plot(p2obj.cell_centers, p2obj.solutionAtTime(10), label='Solution at T=10')
ax.set_ylabel('u(x,t)', rotation=0, labelpad=10)
ax.set_xlabel('x')
ax.set_title('u(x,t) solution; ' + p2obj.slope_limiter_str + ' limiter')
ax.legend(fontsize=8)
fig.tight_layout()
plt.savefig('Figures/problem-2-solution-limiter=' + p2obj.slope_limiter_str + '.png', bbox_inches='tight')