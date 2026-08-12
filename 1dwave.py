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
dx = (c/f) * 3 / 256 # spatial discretization length
sigma = .9 # Courant number
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
ax.set_ylim(-1e4, 1e4)   # adjust if needed

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


results = []

for k in range( 1, int(it), 2 ):
    # save (k-2)th field variable, in gpu this only happend in the registers
    uc = ub.copy()
    # iteration step
    for i in range(res):
        ub[i+1] = 2 * ua[i+1] - uc[i+1] + dt**2 * c**2 / dx**2 * ( ua[i+2] - 2 * ua[i+1] + ua[i] )
    # bc 
    # transducer
    # ub[res+1] = ub[res] + dx * rho * d * omega**2 * np.sin( omega * dt * ( k ) ) - dx**2 / 2 / c**2 / dt**2 * ( ub[res] - 2 * ua[res] + uc[res] )
    d33 = 289e-12
    V0 = 100
    lp = 0.005
    # ub[res+1] = ub[res] + dx * rho * d33 * V0 / lp * omega**2 * np.sin( omega * dt * ( k ) ) - dx**2 / 2 / c**2 / dt**2 * ( ub[res] - 2 * ua[res] + uc[res] )
    # Calculate physical acceleration of the piezo face
    a_n_b = -d33 * V0 * omega**2 * np.sin(omega * dt * k)
    # Central difference ghost node update
    ub[res+1] = ub[res-1] - 2 * dx * rho * a_n_b
    # other end: wall or symmetry or VOID
    # Dirichlet = reflect with inverse phase
    # ub[0] = 0
    # Neumann = relfelct with same phase
    # ub[0] = ub[1]  + dx**2 / 2 / c**2 / dt**2 * ( ub[1] - 2 * ua[1] + uc[1] )
    # Sommerfeld = absorbtion all ( dp / dt + c dp / dx = 0 )
    # ub[0] = ua[0] + dt * c / dx * ( ua[1] - ua[0] ) # this works
    # ub[0] = 1 / ( 1 + dt*c/dx ) * ( ua[0] + dt*c/dx*ub[1] ) # this works, but NOT perfect
    # Impedance = absorbe some ( dp / dt + Z*c dp / dx = 0 )
    ub[0] = ua[0] + Z * dt * c / dx * ( ua[1] - ua[0] ) 
    # ub[0] = 1 / ( 1/dt - c/dx ) * ( ua[0] / dt - c * ub[1] / dx )

    # save (k-2)th field variable, in gpu this only happend in the registers
    uc = ua.copy()
    # iteration step
    for i in range(res):
        ua[i+1] = 2 * ub[i+1] - uc[i+1] + dt**2 * c**2 / dx**2 * ( ub[i+2] - 2 * ub[i+1] + ub[i] )
    # bc 
    # transducer
    # ua[res+1] = ua[res] + dx * rho * d * omega**2 * np.sin( omega * dt * ( k + 1 ) ) - dx**2 / 2 / c**2 / dt**2 * ( ua[res] - 2 * ub[res] + uc[res] )
    d33 = 289e-12
    V0 = 100
    lp = 0.005
    # ua[res+1] = ua[res] + dx * rho * d33 * V0 / lp * omega**2 * np.sin( omega * dt * ( k + 1 ) ) - dx**2 / 2 / c**2 / dt**2 * ( ua[res] - 2 * ub[res] + uc[res] )
    # Calculate physical acceleration of the piezo face
    a_n_a = -d33 * V0 * omega**2 * np.sin(omega * dt * (k + 1))
    
    # Central difference ghost node update
    ua[res+1] = ua[res-1] - 2 * dx * rho * a_n_a
    # ua[res+1] = ua[res] + dx * rho * d * omega**2 * np.sin( omega * dt * ( k + 1 ) ) - dx**2 / 2 / c**2 / dt**2 * ( ua[res] - 2 * ub[res] + uc[res] )
    # other end: wall or symmetry or VOID
    # Dirichlet = reflect with inverse phase
    # ua[0] = 0
    # Neumann = relfelct with same phase
    # ua[0] = ua[1]  + dx**2 / 2 / c**2 / dt**2 * ( ua[1] - 2 * uc[1] + uc[1] )
    # Sommerfeld = absorbtion all ( dp / dt + c dp / dx = 0 )
    # ua[0] = 1 / ( 1/dt - c/dx ) * ( ub[0] / dt - c * ua[1] / dx )
    # ua[0] = ub[0] + dt * c / dx * ( ub[1] - ub[0] ) # this works
    # ua[0] = 1 / ( 1 + dt*c/dx ) * ( ub[0] + dt*c/dx*ua[1] ) # this works, but NOT perfect
    # Impedance = absorbe some ( dp / dt + Z*c dp / dx = 0 )
    ua[0] = ub[0] + dt * Z * c / dx * ( ub[1] - ub[0] )
    if snapshot and k % snap_mod == 0:
        results.append(ua.copy())
        line.set_ydata(ua)
        ax.set_title(f"k = {k}, max = {np.max(ua)}")
        plt.draw()
        plt.pause(0.01)          # redraw
 
 

def update(frame):
    y = results[frame]
    line.set_data(x, y)
    ax.set_title(f"frame = {frame}, max = {np.max(y):.2e}")
    return line,

ani = FuncAnimation(
    fig,
    update,
    frames=len(results), 
    interval=100,
    blit=False
)

ani.save("simulation1d-cdm.mp4", writer="ffmpeg", fps=60, dpi=200)
