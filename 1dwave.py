import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.ion()

# ----- material and bc parameters -----
rho = 1e3 # density
c = 1500 # speed of sound in water
f = 20e3 # transducer frequency
d = 10e-6 # transducer oscillation amplitude in [m]
omega = 2 * 3.14159265359 * f
# ----- simulation parameters -----
it = 50000 # iteration step number
res = 256 # number of pixels
size = res**2
dx = (c/f) * 3 / 256 # spatial discretization length
sigma = .9 # Courant number
dt = sigma * dx / c # time discretization, with CFL criterion
print('dt from CFL: ', dt)
# TODO: examine effects of transducer frequency on max dt
# dt = min(dt, 1/20e4)
# print('dt: ',dt)
snapshot = 1 # if 1 take snapshot
snap_mod = 1 # take snapshot if k % snap_mod == 0

# init field variable arrays
ua = np.zeros([res + 2])
ub = np.zeros([res + 2])
uc = np.zeros([res + 2])
va = np.zeros([res])
vb = np.zeros([res])

# create figure
fig, ax = plt.subplots(figsize=(9, 6))
ax.invert_yaxis()
ax.set_xlabel('')
ax.set_ylabel('')

x = np.arange(res + 2) * dx

fig, ax = plt.subplots(figsize=(9,6))
line, = ax.plot(x, ua)

ax.set_xlabel("x")
ax.set_ylabel("u")
ax.set_ylim(-1e7, 1e7)   # adjust if needed

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

for k in range( 1, int(it/2) ):
    # save (k-2)th field variable, in gpu this only happend in the registers
    uc = ub.copy()
    # iteration step
    for i in range(res):
        ub[i+1] = 2 * ua[i+1] - uc[i+1] + dt**2 * c**2 / dx**2 * ( ua[i+2] - 2 * ua[i+1] + ua[i] )
    # bc 
    # transducer
    ub[res+1] = ub[res] + dx * rho * d * omega**2 * np.sin( omega * dt * ( k ) ) - dx**2 / 2 / c**2 / dt**2 * ( ub[res] - 2 * ua[res] + uc[res] )
    # other end: wall or symmetry or VOID
    # Dirichlet = reflect with inverse phase
    ub[0] = 0
    # Neumann = relfelct with same phase
    # ub[0] = ub[1]
    # Sommerfeld = absorbtion all ( dp / dt + c dp / dx = 0 )
    # Impedance = absorbe some ( dp / dt + Z*c dp / dx = 0 )
    # ub[0] = 1 / ( 1/dt - c/dx ) * ( ua[0] / dt - c * ub[1] / dx )

    # save (k-2)th field variable, in gpu this only happend in the registers
    uc = ua.copy()
    # iteration step
    for i in range(res):
        ua[i+1] = 2 * ub[i+1] - uc[i+1] + dt**2 * c**2 / dx**2 * ( ub[i+2] - 2 * ub[i+1] + ub[i] )
    # bc 
    # transducer
    ua[res+1] = ua[res] + dx * rho * d * omega**2 * np.sin( omega * dt * ( k + 1 ) ) - dx**2 / 2 / c**2 / dt**2 * ( ua[res] - 2 * ub[res] + uc[res] )
    # other end: wall or symmetry or VOID
    # Dirichlet = reflect with inverse phase
    ua[0] = 0
    # Neumann = relfelct with same phase
    # ua[0] = ua[1];
    # Sommerfeld = absorbtion all ( dp / dt + c dp / dx = 0 )
    # ua[0] = 1 / ( 1/dt - c/dx ) * ( ub[0] / dt - c * ua[1] / dx )
    # Impedance = absorbe some ( dp / dt + Z*c dp / dx = 0 )
    if snapshot and k % snap_mod == 0:
        line.set_ydata(ua)
        ax.set_title(f"k = {k}, max = {np.max(ua)}")
        plt.draw()
        plt.pause(0.01)          # redraw
 
 

