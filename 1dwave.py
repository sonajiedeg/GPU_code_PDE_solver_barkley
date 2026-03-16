import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.ion()

it = 50 # iteration step number
res = 256 # number of pixels
size = res**2
rho = 1e3 # density
f = 20e3 # transducer frequency
c = 1500 # speed of sound in water
dx = (c/f) * 3 / 256 # spatial discretization length
sigma = .01 # Courant number
dt = sigma * dx / c # time discretization, with CFL criterion
print('dt from CFL: ', dt)
dt = min(dt, 1/20e4) / 4
print('dt: ',dt)
snapshot = 0 # if 1 take snapshot
snap_mod = 80 # take snapshot if k % snap_mod == 0
d = 10e-6
omega = 2 * 3.14159265359 * f

ua = np.zeros([res + 2])
ub = np.zeros([res + 2])
va = np.zeros([res])
vb = np.zeros([res])

# Create figure
fig, ax = plt.subplots(figsize=(9, 6))
ax.invert_yaxis()
ax.set_xlabel('')
ax.set_ylabel('')

x = np.arange(res + 2) * dx

fig, ax = plt.subplots(figsize=(9,6))
line, = ax.plot(x, ua)

ax.set_xlabel("7")
ax.set_ylabel("u")
ax.set_ylim(-1e7, 1e7)   # adjust if needed

for k in range(int(it/2)):
    # iteration step
    for i in range(res):
        ub[i+1] = ua[i+1] + dt * va[i]
        vb[i] = va[i] + dt * c**2 * ( ua[i+2] - 2 * ua[i+1] + ua[i] ) / dx / dx
    # bc 
    # transducer
    ub[res+1] = ub[res] + dx * rho * d * omega * omega * np.sin( omega * dt * (2 * k) )
    # other end: wall or symmetry or VOID
    # Dirichlet = reflect with inverse phase
    ub[0] = 0;
    # Neumann = relfelct with same phase
    # ub[0] = ub[1];
    # Sommerfeld = absorbtion all ( dp / dt + c dp / dx = 0 )
    # Impedance = absorbe some ( dp / dt + Z*c dp / dx = 0 )
    # ub[0] = 1 / ( 1/dt - c/dx ) * ( ua[0] / dt - c * ub[1] / dx )

    # iteration step
    for i in range(res):
        ua[i+1] = ub[i+1] + dt * vb[i]
        va[i] = vb[i] + dt * c**2 * ( ub[i+2] - 2 * ub[i+1] + ub[i] ) / dx / dx
    # bc 
    # transducer
    ua[res+1] = ua[res] + dx * rho * d * omega * omega * np.sin( omega * dt * (2 * k + 1) )
    # other end: wall or symmetry or VOID
    # Dirichlet = reflect with inverse phase
    ua[0] = 0;
    # Neumann = relfelct with same phase
    # ua[0] = ua[1];
    # Sommerfeld = absorbtion all ( dp / dt + c dp / dx = 0 )
    # ua[0] = 1 / ( 1/dt - c/dx ) * ( ub[0] / dt - c * ua[1] / dx )
    # Impedance = absorbe some ( dp / dt + Z*c dp / dx = 0 )
    if snapshot and k % snap_mod == 0:

        line.set_ydata(ua)

        ax.set_title(f"k = {k}, max = {np.max(ua)}")

        plt.draw()
        plt.pause(0.001)          # redraw
 
 

