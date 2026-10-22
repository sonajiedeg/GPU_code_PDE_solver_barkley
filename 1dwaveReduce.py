import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.ion()

# ----- material and bc parameters -----
rho = 1e3 # density
c = 1500 # speed of sound in water
f = 20e3 # transducer frequency
d = 1e-6 # transducer oscillation amplitude in [m]
omega = 2 * 3.14159265359 * f
# ----- simulation parameters -----
it = 50000 # iteration step number
res = 256 # number of pixels
size = res**2
dx = (c/f) * 1 / 256 # spatial discretization length
sigma = .5 # Courant number
dt = sigma * dx / c # time discretization, with CFL criterion
Z = 1 # impedance BC parameter
'''
    Z = 1: Sommerfeld, no reflection
    Z --> 0: Dirichlet, full reflection, phase inverting bc
    Z --> inf: Neumann, full reflection
'''
print('dt from CFL: ', dt)
# TODO: examine effects of transducer frequency on max dt
# dt = min(dt, 1/20e4)
# print('dt: ',dt)
snapshot = 1 # if 1 take snapshot
snap_mod = 3 # take snapshot if k % snap_mod == 0

# init field variable arrays
ua = np.zeros([res + 2])
ub = np.zeros([res + 2])
uc = np.zeros([res + 2])
va = np.zeros([res])
vb = np.zeros([res])

x = np.arange(res + 2) * dx

# create figure
fig, ax = plt.subplots(figsize=(9,6))
line, = ax.plot(x, ua)

ax.set_xlabel("x")
ax.set_ylabel("u")
ax.set_ylim(-80e3, 80e3)   # adjust if needed

# one step with first order approx
k = 0
for i in range(res):
    ua[i+1] = ub[i+1] + dt * vb[i]
    va[i] = vb[i] + dt * c**2 * ( ub[i+2] - 2 * ub[i+1] + ub[i] ) / dx / dx # this is not really needed in this case
# bc 
# transducer
ua[res+1] = ua[res] + dx * rho * d * omega * omega * np.sin( omega * dt * (2 * k + 1) )
# other end: wall or symmetry or VOID
# Dirichlet = reflect with inverse phase
ua[0] = 0
# haha, all is still 0 after the first step :)

Pa = 75000
bcC1 = 1 / ( 1.0 + dt * c / dx );
bcC2 = Pa * dt * bcC1 * omega;
bcC3 = dt * c / dx * bcC1;


results = []

for k in range( 1, int(it), 2 ):
    # save (k-2)th field variable, in gpu this only happend in the registers
    uc = ub.copy()
    # iteration step
    for i in range(res):
        ub[i+1] = 2 * ua[i+1] - uc[i+1] + dt**2 * c**2 / dx**2 * ( ua[i+2] - 2 * ua[i+1] + ua[i] )
    # bc 
    # transducer
    ub[res+1] = bcC1 * ua[res+1] + bcC3 * ub[res] + bcC2 * np.cos( (omega+2*np.pi*200) * dt * (k) ); # east
    ub[0] = bcC1 * ua[0] + bcC3 * ub[1] + bcC2 * np.cos( omega * dt * (k) ); # east
    # ub[0] = 1/(1+c*dt/dx)* ( ua[0] + omega*Pa*dt*np.cos(omega*dt*k) + c*dt/dx*ub[1])
    # ub[res+1] = 1/(1+c*dt/dx)* ( ua[res+1] + omega*Pa*dt*np.cos(omega*dt*k) + c*dt/dx*ub[res])

    # save (k-2)th field variable, in gpu this only happend in the registers
    uc = ua.copy()
    # iteration step
    for i in range(res):
        ua[i+1] = 2 * ub[i+1] - uc[i+1] + dt**2 * c**2 / dx**2 * ( ub[i+2] - 2 * ub[i+1] + ub[i] )
    # bc 
    # transducer
    ua[res+1] = bcC1 * ub[res+1] + bcC3 * ua[res] + bcC2 * np.cos( (omega+2*np.pi*200)* dt * (k+1) ); # east
    ua[0] = bcC1 * ub[0] + bcC3 * ua[1] + bcC2 * np.cos( omega * dt * (k+1) ); # east
    # ua[0] = 1/(1+c*dt/dx)* ( ub[0] + omega*Pa*dt*np.cos(omega*dt*(k+1)) + c*dt/dx*ua[1])
    # ua[res+1] = 1/(1+c*dt/dx)* ( ub[res+1] + omega*Pa*dt*np.cos(omega*dt*(k+1)) + c*dt/dx*ua[res])


    if snapshot and k % snap_mod == 0:
        results.append(ua.copy())
        line.set_ydata(ua)
        ax.set_title(f"k = {k}, max = {np.max(ua)}")
        plt.draw()
        plt.pause(0.01)          # redraw
