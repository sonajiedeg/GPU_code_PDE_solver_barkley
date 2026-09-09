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
it = 8000 # iteration step number
res = 256 # number of pixels
size = res**2
dx = (c/f) * 3 / 256 # spatial discretization length
sigma = .9 # Courant number
dt = sigma * dx / c # time discretization, with CFL criterion
Z = .8 # impedance BC parameter
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
# ----- Sponge layer parameters -----


# ----- PML parameters -----
spongeLayerDepth = 128
res = res + spongeLayerDepth

# Target a theoretical reflection coefficient of 0.0001 (-80dB)
R_target = 1e-4
# Industry standard max sigma based on layer depth and wave speed
sigma_max = - (3 * c) / (2 * spongeLayerDepth * dx) * np.log(R_target)
# sigma_max = 0

# Quadratic sigma profile (0 at the interface, max at the boundary)
sigma = np.array([sigma_max * ((i - spongeLayerDepth) / spongeLayerDepth)**2 for i in range(spongeLayerDepth)])
sigma_dt = sigma * dt

# Updated True PML coefficients
xi = (2 - sigma_dt**2) / (1 + sigma_dt) # The missing sigma^2 term is accounted for here
eta = (sigma_dt - 1) / (1 + sigma_dt)
zeta = (c**2 * dt**2 / dx**2) / (1 + sigma_dt)

# Define ramp duration (e.g., 3 full wave periods)
ramp_periods = 12.0

'''
spongeLayerDepth = 256 
res = res + spongeLayerDepth
b1 = 0. # damping coefficient at the boundary between the sponge and sim space
b0 = c**2 / 16 # damping coefficient at the boundary of the sponge and nothing
# b = [0] * spongeLayerDepth
b2 = [ i / spongeLayerDepth * b1 + ( 1 - i / spongeLayerDepth ) * b0 for i in range(0,spongeLayerDepth) ]
b3 = [ b0 * np.exp( -i / ( spongeLayerDepth / 4 ) ) for i in range(spongeLayerDepth) ]
b4 = [ b0 * ( 1/np.pi * np.arctan( (-i + spongeLayerDepth / 2)/4 ) + 1/2 ) for i in range(spongeLayerDepth) ]
b5 = [ b0 * (i-spongeLayerDepth)**4/spongeLayerDepth**4 for i in range(spongeLayerDepth) ]
b = np.array( b5 )
xi = 2 / ( 1 + b * dt / 2 )
eta = ( b * dt / 2 - 1 ) / ( 1 + b * dt / 2 )
zeta = c**2 * dt**2 / dx**2 / ( 1 + b * dt / 2 )
'''

# print(b)
# print(xi)
# print(eta)
# print(zeta)

'''
idx = range(0,spongeLayerDepth,1)
plt.figure(13)
plt.plot(idx,b2,"r-")
plt.plot(idx,b3,"g-")
plt.plot(idx,b4,"b-")
plt.plot(idx,b5,"k-")
plt.show()
'''


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
ax.set_ylim(-1e7, 1e7)   # adjust if needed

# one step with first order approx
k = 0
'''
for i in range(0,spongeLayerDepth):
    ua[i+1] = ub[i+1] + dt * vb[i]
    va[i] = vb[i] - dt * b[i] * ub[i] + dt * c**2 * ( ub[i+2] - 2 * ub[i+1] + ub[i] ) / dx / dx # this is not really needed in this case
'''
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
    for i in range(0,spongeLayerDepth-1):
        ub[i+1] = xi[i+1] * ua[i+1] + eta[i+1] * uc[i+1] + zeta[i+1] * ( ua[i+2] - 2 * ua[i+1] + ua[i] )
    for i in range(spongeLayerDepth-1,res):
        ub[i+1] = 2 * ua[i+1] - uc[i+1] + dt**2 * c**2 / dx**2 * ( ua[i+2] - 2 * ua[i+1] + ua[i] )
    # bc 
    # transducer
    # ub[res+1] = ub[res] + dx * rho * d * omega**2 * np.sin( omega * dt * ( k ) ) - dx**2 / 2 / c**2 / dt**2 * ( ub[res] - 2 * ua[res] + uc[res] )
    # Apply the ramp to your transducer source in your update loop:
    t_current = k * dt
    ramp = 1.0 if (f * t_current) > ramp_periods else 0.5 * (1 - np.cos(np.pi * (f * t_current) / ramp_periods))
    ub[res+1] = ub[res] + dx * rho * d * omega**2 * np.sin(omega * dt * ( k ) ) * ramp - dx**2 / 2 / c**2 / dt**2 * ( ub[res] - 2 * ua[res] + uc[res] )
    # other end: wall or symmetry or VOID
    # Dirichlet = reflect with inverse phase
    # ub[0] = 0
    # Neumann = relfelct with same phase
    # ub[0] = ub[1]  + dx**2 / 2 / c**2 / dt**2 * ( ub[1] - 2 * ua[1] + uc[1] )
    # Sommerfeld = absorbtion all ( dp / dt + c dp / dx = 0 )
    ub[0] = ua[0] + dt * c / dx * ( ua[1] - ua[0] ) # this works
    # ub[0] = 1 / ( 1 + dt*c/dx ) * ( ua[0] + dt*c/dx*ub[1] ) # this works, but NOT perfect
    # Impedance = absorbe some ( dp / dt + Z*c dp / dx = 0 )
    # ub[0] = ua[0] + Z * dt * c / dx * ( ua[1] - ua[0] ) 
    # ub[0] = 1 / ( 1/dt - c/dx ) * ( ua[0] / dt - c * ub[1] / dx )

    # save (k-2)th field variable, in gpu this only happend in the registers
    uc = ua.copy()
    # iteration step
    for i in range(0,spongeLayerDepth-1):
        ua[i+1] = xi[i+1] * ub[i+1] + eta[i+1] * uc[i+1] + zeta[i+1] * ( ub[i+2] - 2 * ub[i+1] + ub[i] )
    for i in range(spongeLayerDepth-1,res):
        ua[i+1] = 2 * ub[i+1] - uc[i+1] + dt**2 * c**2 / dx**2 * ( ub[i+2] - 2 * ub[i+1] + ub[i] )
    # bc 
    # transducer
    # ua[res+1] = ua[res] + dx * rho * d * omega**2 * np.sin( omega * dt * ( k + 1 ) ) - dx**2 / 2 / c**2 / dt**2 * ( ua[res] - 2 * ub[res] + uc[res] )
    # Apply the ramp to your transducer source in your update loop:
    t_current = ( k + 1 ) * dt
    ramp = 1.0 if (f * t_current) > ramp_periods else 0.5 * (1 - np.cos(np.pi * (f * t_current) / ramp_periods))
    ua[res+1] = ua[res] + dx * rho * d * omega**2 * np.sin( omega * dt * ( k + 1 ) ) * ramp - dx**2 / 2 / c**2 / dt**2 * ( ua[res] - 2 * ub[res] + uc[res] )
    # other end: wall or symmetry or VOID
    # Dirichlet = reflect with inverse phase
    # ua[0] = 0
    # Neumann = relfelct with same phase
    # ua[0] = ua[1]  + dx**2 / 2 / c**2 / dt**2 * ( ua[1] - 2 * uc[1] + uc[1] )
    # Sommerfeld = absorbtion all ( dp / dt + c dp / dx = 0 )
    # ua[0] = 1 / ( 1/dt - c/dx ) * ( ub[0] / dt - c * ua[1] / dx )
    ua[0] = ub[0] + dt * c / dx * ( ub[1] - ub[0] ) # this works
    # ua[0] = 1 / ( 1 + dt*c/dx ) * ( ub[0] + dt*c/dx*ua[1] ) # this works, but NOT perfect
    # Impedance = absorbe some ( dp / dt + Z*c dp / dx = 0 )
    # ua[0] = ub[0] + dt * Z * c / dx * ( ub[1] - ub[0] )
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
