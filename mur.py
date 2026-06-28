import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from wave_eqn2d import WaveEqn2D



h = 0.001 / 256 / 32
c = 1500
s = 0.4
dt = h * s / c
T = 60 
freq = 2 * np.pi / T
nx = ny = 256
rho = 1e3
d = 10e-6
A = h * rho * d * ( freq )**2
sim = WaveEqn2D(nx, ny, c=c, h=h, dt=dt, use_mur_abc=True)
lim = 3e-12
div = 5
top = div-1

fig, ax = plt.subplots()
ax.axis("off")
# img = ax.imshow(sim.u[0], vmin=0, vmax=40, cmap='Blues_r')
img = ax.imshow(sim.u[0], vmin=-lim, vmax=lim, cmap='Blues_r')

def update(i):
    """Advance the simulation by one tick."""
    # A regular sinusoidal signal at the centre of the domain.
    # sim.u[0, ny//2, nx//8] = A * np.sin(i * freq)
    sim.u[0, ny//4:3*ny//4, nx-1] = sim.u[0, ny//4:3*ny//4, nx-2] + A / h * np.sin(i * freq) # 2
    # sim.u[0, top*ny//div:ny-1, nx-1] = sim.u[0, top*ny//div:ny-1, nx-2] + A * np.sin(i * freq) # 3
    # sim.u[0, 0, top*nx//div:nx-1] = sim.u[0, 1, top*nx//div:nx-1] + A * np.sin(i * freq) # 3 
    # sim.u[0, ny//4:3*ny//4, nx-1] = A * np.sin(i * freq)**2
    # sim.u[0, ny//4:3*ny//4, 0] = sim.u[0, ny//4:3*ny//4, 1] - A / h * np.sin(i * freq * 2 + np.pi/2 ) # 2
    # sim.u[0, 0:ny//div, 0] = sim.u[0, 0:ny//div, 1] + A * np.sin(i * freq ) # 3
    # sim.u[0, ny-1, 0:nx//div] = sim.u[0, ny-2, 0:nx//div] + A * np.sin(i * freq ) # 3 
    # sim.u[0, ny//4:3*ny//4, 0] = A * np.sin(i * freq * 2 + np.pi/2 )**2
    sim.update()

def init():
    """
    Initialization, because we're blitting and need references to the
    animated objects.
    """
    return img,

def animate(i):
    """Draw frame i of the animation."""
    update(i)
    img.set_data(sim.u[0])
    return img,

interval, nframes = sim.dt, 20000
ani = animation.FuncAnimation(fig, animate, frames=nframes,
repeat=False,
init_func=init, interval=interval, blit=True)
plt.show()
