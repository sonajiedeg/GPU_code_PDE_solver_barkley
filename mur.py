import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from wave_eqn2d import WaveEqn2D

A = 80
dt = 4
T = 60
freq = 2 * np.pi / T
nx = ny = 256
sim = WaveEqn2D(nx, ny, dt=dt, use_mur_abc=True)

fig, ax = plt.subplots()
ax.axis("off")
img = ax.imshow(sim.u[0], vmin=0, vmax=40, cmap='Blues_r')

def update(i):
    """Advance the simulation by one tick."""
    # A regular sinusoidal signal at the centre of the domain.
    sim.u[0, ny//2, nx//8] = A * np.sin(i * freq)
    # sim.u[0, ny//4:3*ny//4, nx-1] = A * np.sin(i * freq)
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

interval, nframes = sim.dt, 2000
ani = animation.FuncAnimation(fig, animate, frames=nframes,
repeat=False,
init_func=init, interval=interval, blit=True)
plt.show()
