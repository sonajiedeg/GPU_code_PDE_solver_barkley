import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
LABEL = 12
FONT = 18
res = 256
dt = 0.0076893502846360206604003906250000000000
STEP = 20.0

dir = r"/mnt/c/Users/Gedei Janos Jonatan/Dokumentumok/LaTeX Projects/MSc-repo/3/SonoChem/GPU_code_PDE_solver_barkley/"
s1 = r"u_snap.txt"

data = np.loadtxt(dir + s1)

num_frames = data.size // ( res * res )
data = data.reshape((num_frames, res, res))

vmin_value = -0.5
vmax_value = 0.5

# Create figure
fig, ax = plt.subplots(figsize=(9, 6))

heatmap = ax.imshow(
    data[0],
    cmap='winter',
    interpolation='nearest',
    vmin=vmin_value,
    vmax=vmax_value
)

ax.invert_yaxis()
ax.set_xlabel('')
ax.set_ylabel('')

plt.tick_params(axis='both', which='major', labelsize=30)

colorbar = plt.colorbar(heatmap)
colorbar.ax.tick_params(labelsize=30)

# Animation update function
def update(frame):
    heatmap.set_data(data[frame])
    ax.set_title(f"simulation time {(frame*dt*STEP):.4f}", fontsize=20)
    return [heatmap]

# Create animation
ani = FuncAnimation(
    fig,
    update,
    frames=num_frames,
    interval=50,   # ms between frames
    blit=True
)

ani.save("simulation.mp4", writer="ffmpeg", fps=30)

plt.show()
