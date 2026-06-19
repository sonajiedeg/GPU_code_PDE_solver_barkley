import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. CONFIGURATION ---
LABEL = 12
FONT = 18
res = 256
dt = 5.859374999999999e-09 / 4
STEP = 500.0

# Choose your mode: 'heatmap_anim', 'line_anim', or 'line_static'
RUN_MODE = 'heatmap_anim'
# RUN_MODE = 'line_anim'
# RUN_MODE = 'line_static'

# Parameters for the new 1D features
TARGET_AXIS = 'col'     # Choose 'row' or 'col'
# TARGET_AXIS = 'row'     # Choose 'row' or 'col'
TARGET_INDEX = 128      # Which specific row or column index to extract (0 to 255)
FRAMES_TO_PLOT = [0, 50, 100, 150, 200] # Used only for 'line_static' mode


# --- 2. DATA LOADING ---
print("Loading data...")
data = np.fromfile("u_snap.bin", dtype=np.float32)

num_frames = data.size // (res * res)
data = data.reshape((num_frames, res, res))

print("data size:", data.size)
print("expected per frame:", res*res)
print("num_frames:", num_frames)


# --- 3. MODES ---

if RUN_MODE == 'heatmap_anim':
    # ---------------------------------------------------------
    # ORIGINAL FEATURE: 2D Heatmap Animation
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))

    heatmap = ax.imshow(
        data[0],
        cmap='winter',
        interpolation='nearest'
    )
    
    ax.invert_yaxis()
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.tick_params(axis='both', which='major', labelsize=8)

    def update_heatmap(frame):
        heatmap.set_data(data[frame])
        heatmap.set_clim(np.min(data[frame]), np.max(data[frame]))
        ax.set_title(f"Simulation time {(frame*dt*STEP):.4f}, frame {frame}", fontsize=20)
        return [heatmap]

    ani = FuncAnimation(fig, update_heatmap, frames=num_frames, interval=20)
    ani.save("sCFL_0_6_30lam_ref.mp4", writer="ffmpeg", fps=36, dpi=200)
    plt.show()

elif RUN_MODE == 'line_anim':
    # ---------------------------------------------------------
    # NEW FEATURE 1: 1D Line Plot Animation (Row or Column)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    x_data = np.arange(res)

    # Initial data setup
    if TARGET_AXIS == 'row':
        y_data = data[0, TARGET_INDEX, :]
        title_str = f"Row {TARGET_INDEX}"
        # Find global min/max for this specific row across ALL frames to lock Y-axis
        global_min = np.min(data[:, TARGET_INDEX, :])
        global_max = np.max(data[:, TARGET_INDEX, :])
    else:
        y_data = data[0, :, TARGET_INDEX]
        title_str = f"Column {TARGET_INDEX}"
        # Find global min/max for this specific column across ALL frames to lock Y-axis
        global_min = np.min(data[:, :, TARGET_INDEX])
        global_max = np.max(data[:, :, TARGET_INDEX])

    # Plot initial line
    line, = ax.plot(x_data, y_data, lw=2, color='blue')

    # Formatting
    ax.set_xlim(0, res - 1)
    # Add a 10% buffer to the y-axis limits for better viewing
    y_range = global_max - global_min
    ax.set_ylim(global_min - 0.1 * y_range, global_max + 0.1 * y_range)
    
    ax.set_xlabel("Cell Number", fontsize=FONT)
    ax.set_ylabel("Amplitude", fontsize=FONT)
    ax.tick_params(axis='both', which='major', labelsize=LABEL)
    ax.grid(True, linestyle='--', alpha=0.6)

    def update_line(frame):
        if TARGET_AXIS == 'row':
            line.set_ydata(data[frame, TARGET_INDEX, :])
        else:
            line.set_ydata(data[frame, :, TARGET_INDEX])
            
        ax.set_title(f"1D Cut ({title_str}) | Time {(frame*dt*STEP):.4f} | Frame {frame}", fontsize=16)
        return [line]

    ani = FuncAnimation(fig, update_line, frames=num_frames, interval=20)
    
    # Save with a descriptive filename
    save_name = f"1D_anim_{TARGET_AXIS}_{TARGET_INDEX}.mp4"
    print(f"Saving animation to {save_name}...")
    ani.save(save_name, writer="ffmpeg", fps=36, dpi=200)
    plt.show()

elif RUN_MODE == 'line_static':
    # ---------------------------------------------------------
    # NEW FEATURE 2: Static Multiple Lines (Selected Frames)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    x_data = np.arange(res)
    
    # Filter out any frames that might exceed our actual data length
    valid_frames = [f for f in FRAMES_TO_PLOT if f < num_frames]

    for frame in valid_frames:
        if TARGET_AXIS == 'row':
            y_data = data[frame, TARGET_INDEX, :]
            title_str = f"Row {TARGET_INDEX}"
        else:
            y_data = data[frame, :, TARGET_INDEX]
            title_str = f"Column {TARGET_INDEX}"
            
        # Plot each frame as a separate line with a label for the legend
        ax.plot(x_data, y_data, lw=2, label=f"Frame {frame} (t={(frame*dt*STEP):.4f})")

    # Formatting
    ax.set_xlim(0, res - 1)
    ax.set_xlabel("Cell Number", fontsize=FONT)
    ax.set_ylabel("Amplitude", fontsize=FONT)
    ax.set_title(f"1D Profiles Over Time ({title_str})", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=LABEL)
    
    # Add legend and grid
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    # Uncomment the line below if you want to automatically save the static plot as an image
    # plt.savefig(f"1D_static_{TARGET_AXIS}_{TARGET_INDEX}.png", dpi=200)
    plt.show()

else:
    print("Invalid RUN_MODE selected. Please check your configuration.")
