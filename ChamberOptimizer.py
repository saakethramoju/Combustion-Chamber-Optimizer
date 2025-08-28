from rocketcea.cea_obj_w_units import CEA_Obj
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Inputs:
fuel        = 'RP-1'
oxidizer    = 'LOX'
Pc          = 400        # chamber pressure, psia
Pe          = 8          # exit pressure, psia
thrust      = 3500       # thrust, lbf
MR          = 2.3        # mixture ratio
Dc          = [3, 5]     # chamber diameter range, in
Lc          = [1, 20]    # chamber cylindrical length range, in
eps_c       = [2, 4.3]   # contractio ratio range
theta_c     = [35, 40]   # contraction angle range, °
Rc_Rt       = [0.8, 1.5] # convergence radius factor
Rtc_Rt      = [0.8, 1.5] # lead-in factor
L_star      = [40, 50]   # characteristic length range, in
plot        = False      # True if you want to plot the contour


# ----------------------------------------------------------------------------- #

fullscale = CEA_Obj(oxName=oxidizer, fuelName=fuel, temperature_units='degK', 
                 cstar_units='m/sec', specific_heat_units='kJ/kg degK', 
                 sonic_velocity_units='m/s', enthalpy_units='J/kg', 
                 density_units='kg/m^3')

eps = fullscale.get_eps_at_PcOvPe(Pc, MR, Pc/Pe, 1, 1)
Me = fullscale.get_MachNumber(Pc, MR, eps, 1, 1)
_, _, ae = fullscale.get_SonicVelocities(Pc, MR, eps, 1, 1)
Ve = Me * ae
mdot = (thrust * 4.44822) / Ve
Pt = Pc / fullscale.get_Throat_PcOvPe(Pc, MR)
mwt, gammat = fullscale.get_Throat_MolWt_gamma(Pc, MR, eps, 1)
_, Tt, _ = fullscale.get_Temperatures(Pc, MR, eps, 1, 1)
At = (mdot / (Pt * 6894.76)) * np.sqrt(8314.46261815324 * Tt / (mwt * gammat)) * 1550
Rt = np.sqrt(At/(np.pi))


def contour(Rt, Lc, eps_c, theta_c, Rc_Rt, Rtc_Rt):
    Rc = Rc_Rt * Rt # in
    Rtc = Rtc_Rt * Rt # in

    # Chamber cylinder section
    theta_c = np.deg2rad(theta_c)
    f1z = np.linspace(0, Lc, 100)
    f1r = np.ones(len(f1z))*np.sqrt(eps_c)*Rt

    # Converging entrance arc
    t = np.linspace(np.pi/2, np.pi/2 - theta_c, 100)
    f2z = Rc*np.cos(t) + Lc
    f2r = Rc*np.sin(t) + Rt*np.sqrt(eps_c) - Rc
    inds = np.argsort(f2z)
    f2z = f2z[inds]
    f2r = f2r[inds]

    # Converging linear section
    y3 = Rt*(np.sqrt(eps_c)-1) - (Rc-Rc*np.cos(theta_c)) - (Rtc-Rtc*np.cos(theta_c))
    x3 = y3 / (np.tan(theta_c))
    f3z = np.linspace(f2z[-1], f2z[-1] + x3, 100)
    m = -np.tan(theta_c)
    f3r = m*(f3z - f2z[-1]) + f2r[-1]

    # Converging throat arc
    t = np.linspace(np.pi + np.pi/2 - theta_c, 3*np.pi/2, 100)
    h = f3z[-1] + Rtc*np.sin(theta_c)
    k = Rt + Rtc
    f4z = Rtc*np.cos(t) + h
    f4r = Rtc*np.sin(t) + k
    inds = np.argsort(f4z)
    f4z = f4z[inds]
    f4r = f4r[inds]
    
    z = np.concatenate((f1z,
                        f2z[1:],
                        f3z[1:], 
                        f4z[1:]))
    r = np.concatenate((f1r,
                        f2r[1:],
                        f3r[1:],
                        f4r[1:]))
    return z, r


def volume(z, r): # same units for z and r
    z = np.array(z)
    r = np.array(r)
    integrand = np.pi * r**2
    V = np.trapezoid(integrand, z)
    return V


def surface_area(z, r): # same units for z and r
    z = np.array(z)
    r = np.array(r)
    dr_dz = np.gradient(r, z)
    integrand = 2 * np.pi * r * np.sqrt(1 + dr_dz**2)
    S = np.trapezoid(integrand, z)
    return S


def get_Lstar(z, r, At): # in
    V = volume(z, r)
    return V / At

# x = [Lc, eps_c, theta_c, Rc_Rt, Rtc_Rt]

def objective(x):
    z, r = contour(Rt, x[0], x[1], x[2], x[3], x[4])
    L = get_Lstar(z, r, At)
    S = surface_area(z, r)
    return S / L

def min_eps_c_constraint(x):
    Ac = x[1]*At
    D = np.sqrt(Ac / (np.pi/4))
    return D - Dc[0]

def max_eps_c_constraint(x):
    Ac = x[1]*At
    D = np.sqrt(Ac / (np.pi/4))
    return Dc[1] - D

def min_Lstar_constraint(x):
    z, r = contour(Rt, x[0], x[1], x[2], x[3], x[4])
    L = get_Lstar(z, r, At)
    return L - L_star[0]

def max_Lstar_constraint(x):
    z, r = contour(Rt, x[0], x[1], x[2], x[3], x[4])
    L = get_Lstar(z, r, At)
    return L_star[1] - L

def plot_contour(z, r, plot = True):
    if plot:
        plt.figure
        plt.ylim([0, 5])
        plt.grid(True)
        plt.plot(z, r, linewidth=3)
        plt.show()

bounds = [
    (Lc[0],     Lc[1]),
    (eps_c[0],  eps_c[1]),
    (theta_c[0],theta_c[1]),
    (Rc_Rt[0],  Rc_Rt[1]),
    (Rtc_Rt[0], Rtc_Rt[1])
]

constraints = [
    {"type": "ineq", "fun": min_eps_c_constraint},
    {"type": "ineq", "fun": max_eps_c_constraint},
    {"type": "ineq", "fun": min_Lstar_constraint},
    {"type": "ineq", "fun": max_Lstar_constraint}
]

x0 = [
    np.mean(Lc),
    np.mean(eps_c),
    np.mean(theta_c),
    np.mean(Rc_Rt),
    np.mean(Rtc_Rt)
]

res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)

if not res.success:

    print("Success:", res.success)
    print("Message:", res.message)

else:

    z, r = contour(Rt, res.x[0], res.x[1], res.x[2], res.x[3], res.x[4])
    dc = np.sqrt((At * res.x[1]) / (np.pi/4))

    print(f"Throat Area: {At:.3f} in^2")
    print(f"Throat Radius: {Rt:.3f} in")
    print(f"Chamber Cylindrical Length: {res.x[0]:.3f} in")
    print(f"Chamber Diameter: {dc:.3f} in")
    print(f"Contraction Ratio: {res.x[1]:.3f}")
    print(f"Convergence Angle: {res.x[2]:.3f}°")
    print(f"Convergence Radius Factor: {res.x[3]:.3f}")
    print(f"Lead-in Factor: {res.x[4]:.3f}")
    print(f"L*: {get_Lstar(z, r, At):.3f} in")
    print(f"Mass Flow: {mdot:.3f} kg/s")

    plot_contour(z, r, plot)

