import numpy as np

def getNumericalFluxes(pdeobj, Uprev, Phi):
    '''Accepts: - The numerical solution at t_{n-1}, <Uprev>
                - The limited slopes, <Phi>'''
    '''Returns: - The right fluxes minus the left fluxes'''
    alpha = findMaxCharacteristicSpeed(pdeobj, Uprev)

    deltaU = np.roll(Uprev, -1) - Uprev

    # Calculate right fluxes, F(j+1/2)
    U_tilde_minus = Uprev + Phi*deltaU / 2
    U_tilde_plus = np.roll(Uprev - Phi*deltaU / 2, -1)
    right_fluxes = LaxFriedrichsFlux(pdeobj, U_tilde_minus, U_tilde_plus, alpha)

    # Calculate left fluxes, F(j-1/2)
    U_tilde_minus = np.roll(Uprev + Phi*deltaU / 2, 1)
    U_tilde_plus = Uprev - Phi*deltaU / 2
    left_fluxes = LaxFriedrichsFlux(pdeobj, U_tilde_minus, U_tilde_plus, alpha)

    return right_fluxes - left_fluxes

def LaxFriedrichsFlux(pdeobj, a, b, alpha):
    '''Accepts: - U to the left of a boundary, <a>
                - U to the right of a boundary, <b>
                - the max characteristic speed, <alpha>'''
    '''Returns: - the Lax-Friedrichs numerical flux approximation'''
    return (pdeobj.f(a) + pdeobj.f(b))/2 + alpha*(a - b)/2

def findMaxCharacteristicSpeed(pdeobj, Uprev):
    '''Accepts: - the numerical solution at t_{n-1}, <Uprev>'''
    '''Returns: - max{f'(u)} across all cells'''
    df = pdeobj.f(Uprev) - pdeobj.f(np.roll(Uprev,1))
    du = Uprev - np.roll(Uprev,1)
    nonzero_indices = np.where(du != 0)[0]
    return np.max(np.abs(df[nonzero_indices] / du[nonzero_indices]))

def getLimitedSlopes(pdeobj, Uprev):
    '''Accepts: - the numerical solution at t_{n-1}, <Uprev>'''
    '''Returns: - the limited slopes at each x_j'''

    # Allocate forward/backward difference arrays
    forward = np.zeros((len(Uprev)))
    backward = np.zeros((len(Uprev)))
    
    # Store all of the forward/backward differences
    J = pdeobj.J
    for j in range(1, J):
        forward[j] = Uprev[j+1] - Uprev[j]
        backward[j] = Uprev[j] - Uprev[j-1]

    forward[0] = Uprev[1] - Uprev[0]
    backward[0] = Uprev[0] - Uprev[J]
    forward[J] = Uprev[0] - Uprev[J]
    backward[J] = Uprev[J] - Uprev[J-1]

    # Set denominators to be small if they are zero
    epsilon = 1e-16
    forward[forward == 0] = epsilon

    # Limit the slopes using the slope limiter
    return pdeobj.slope_limiter(backward / forward)

def minmodLimiterVectorized(R): 
    return np.array([max(0, min(1, r)) for r in R])

def superbeeLimiterVectorized(R):
    return np.array([max(0, min(2*r, 1), min(r, 2)) for r in R])