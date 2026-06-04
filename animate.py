import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import SymLogNorm
LABEL = 12
FONT = 18
res = 256
dt = 5.859374999999999e-09 / 4
STEP = 500.0

dir = r"/mnt/c/Users/Gedei Janos Jonatan/Dokumentumok/LaTeX Projects/MSc-repo/3/SonoChem/GPU_code_PDE_solver_barkley/"
s1 = r"u_snap.bin"

# data = np.loadtxt(dir + s1)
data = np.fromfile("u_snap.bin", dtype=np.float32)

num_frames = data.size // ( res * res )
data = data.reshape((num_frames, res, res))

print("data size:", data.size)
print("expected per frame:", res*res)
print("num_frames:", data.size // (res*res))

# vmin_value = -1e1
# vmax_value =  1e1

# Create figure
fig, ax = plt.subplots(figsize=(9, 6))

heatmap = ax.imshow(
    data[0],
    cmap='winter',
    #norm=SymLogNorm(vmin=-2e6, vmax=+2e6, linthresh=1e6), # Set your min > 0 and max
    interpolation='nearest',
    # vmin=vmin_value,
    # vmax=vmax_value
)

ax.invert_yaxis()
ax.set_xlabel('')
ax.set_ylabel('')

plt.tick_params(axis='both', which='major', labelsize=8)

# colorbar = plt.colorbar(heatmap)
# colorbar.ax.tick_params(labelsize=30)

# Animation update function
def update(frame):
    # sub_data = data[frame] - np.mean(data[frame])
    # heatmap.set_data(sub_data)
    heatmap.set_data(data[frame])
    heatmap.set_clim(np.min(data[frame]), np.max(data[frame])) # limit scales to actual min/max
    # heatmap.set_clim(-2e5,2e5) # limit scales to actual min/max
    # heatmap.set_norm(matplotlib.colors.SymLogNorm(vmin=np.min(data[frame]), vmax=np.max(data[frame])))
    ax.set_title(f"simulation time {(frame*dt*STEP):.4f}, frame {frame}", fontsize=20)
    return [heatmap]

# Create animation
ani = FuncAnimation(
    fig,
    update,
    frames=num_frames,
    interval=100,   # ms between frame
)

ani.save("simulation.mp4", writer="ffmpeg", fps=12, dpi=200)

plt.show()

