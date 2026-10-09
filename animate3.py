import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. CONFIGURATION ---
LABEL = 12
FONT = 18
res = 256 
# dt = 5.859374999999999e-09 / 4
# dt = 4.71698115e-07
# dt = 1.17370888e-07
# dt = 5.86166458e-08
# dt = 3.55113627e-08
dt = 7.10227255e-08
STEP = 500.0
p0 = 101325 # Pa

PHASE_SNAP = True 
# Choose your mode: 'heatmap_anim', 'line_anim', 'line_static', or 'point_static'
RUN_MODE = 'heatmap_anim'
# RUN_MODE = 'line_anim'
# RUN_MODE = 'line_anim_phase_snap'
# RUN_MODE = 'line_static'
# RUN_MODE = 'point_static'

# choose .mp4 name
ani_name = 'hmap_name_tmp'

# choose file to read
fname = 'uf25000df200s3.bin'
# fname = 'u_snap_f_44000.00_df_200.00_s_0.bin'
# fname = 'u_snap_f_44000.00_df_200.00_s_8.bin'
# fname = 'u_snap.bin'
# fname = 'u_snap_gradx.bin'
# fname = 'u_snap_grady.bin'

# down sampling settings
DOWN_SAMPLE = True
num_frames_ds_target = 100 

# Parameters for the 1D Spatial features (line modes)
TARGET_AXIS = 'row'     # Choose 'row' or 'col'
# TARGET_INDEX = 128      # Which specific row or column index to extract (0 to 255)
TARGET_INDEX = 64 # Which specific row or column index to extract (0 to 255)
# FRAMES_TO_PLOT = [0, 50, 100, 150, 200] # Used only for 'line_static' mode
FRAMES_TO_PLOT = [] # Used only for 'line_static' mode
title_str = f""
fs = 1
fn = 10
for i in range(num_frames_ds_target-1, num_frames_ds_target-1 - fs * fn, -fs):
    FRAMES_TO_PLOT.append(i)

# FRAMES_TO_PLOT = [1200]
# FRAMES_TO_PLOT.append(1199)

# Parameters for the 1D Temporal feature (point mode)
# List of (row, col) tuples you want to track over time
# POINTS_TO_PLOT = [(256, 256), (256, 138), (64, 256)] 
POINTS_TO_PLOT = [(64, 0), (64, 127)] 
# POINTS_TO_PLOT = [(64, 64), (64, 32), (64, 96)] 


# --- 2. DATA LOADING ---
print("Loading data...")
if PHASE_SNAP:
    # dt = 1.40625014e-06
    # dt = 2.81250027e-06
    # t_end = 0.0072
    # dt = 4.71698115e-07
    # dt = 1.17370888e-07
    # dt = 5.86166458e-08
    # dt = 2.66439297e-08
    # dt = 4.00128037e-08
    # dt = 3.55113627e-08
    # dt = 3.55113627e-08
    dt = 7.10227255e-08
    freq = 44000
    Tper = 1 / freq / 8
    PhaseSnapOffsetDt = ( np.pi * 0/3 ) / 2 / np.pi / freq
    file_size = os.path.getsize(fname)
    frame_bytes = res * res * 4
    num_frames = file_size // frame_bytes
    # print('------------------------->', file_size // frame_bytes == data.size // (res * res))
    # num_frames = data.size // (res * res)
    # data = data.reshape((num_frames, res, res))
    step = num_frames // num_frames_ds_target
    selected_frames = []
    
    phase_counter = 0
    t_current = 0
    frame_needed = False
    with open(fname, "rb") as f:
        for frame_idx in range(num_frames):
            t_current = frame_idx * dt
            t_peak = phase_counter * Tper + PhaseSnapOffsetDt
            if t_current <= t_peak and  t_peak < t_current+dt:
                frame_needed = True
                phase_counter = phase_counter+1
            else:
                frame_needed = False 

            if frame_needed:
                # Read and parse only the required frame
                chunk = f.read(res * res * 4)
                frame = np.frombuffer(chunk, dtype=np.float32).reshape(res, res)
                selected_frames.append(frame)
            else:
                # Skip 24 frames instantly without loading data into RAM
                f.seek(res * res * 4, 1)
    num_frames_ds = len(selected_frames)
    step = num_frames // num_frames_ds
    print("num_frames:", num_frames)
    print('num_frames_ds_target:', num_frames_ds_target)
    print('num_frames_ds:', num_frames_ds)
    print('len(selected_frames):',len(selected_frames) )
    data = np.array(selected_frames).reshape((num_frames_ds, res, res))
    print("data size:", data.size)
    print("expected per frame:", res*res)
    print('step:', step)
    num_frames = num_frames_ds

elif DOWN_SAMPLE:
    file_size = os.path.getsize(fname)
    frame_bytes = res * res * 4
    num_frames = file_size // frame_bytes
    # print('------------------------->', file_size // frame_bytes == data.size // (res * res))
    # num_frames = data.size // (res * res)
    # data = data.reshape((num_frames, res, res))
    step = num_frames // num_frames_ds_target
    selected_frames = []

    with open(fname, "rb") as f:
        for frame_idx in range(num_frames):
            if frame_idx % step == 0:
                # Read and parse only the required frame
                chunk = f.read(res * res * 4)
                frame = np.frombuffer(chunk, dtype=np.float32).reshape(res, res)
                selected_frames.append(frame)
            else:
                # Skip 24 frames instantly without loading data into RAM
                f.seek(res * res * 4, 1)
    num_frames_ds = len(selected_frames)
    step = num_frames // num_frames_ds
    print("num_frames:", num_frames)
    print('num_frames_ds_target:', num_frames_ds_target)
    print('num_frames_ds:', num_frames_ds)
    print('len(selected_frames):',len(selected_frames) )
    data = np.array(selected_frames).reshape((num_frames_ds, res, res))
    print("data size:", data.size)
    print("expected per frame:", res*res)
    print('step:', step)
    num_frames = num_frames_ds
else:
    data = np.fromfile(fname, dtype=np.float32)
    num_frames = data.size // (res * res)
    data = data.reshape((num_frames, res, res))
    print("num_frames:", num_frames)
    print("data size:", data.size)
    print("expected per frame:", res*res)


# add ambient pressure
data = data

print("-------------------------------->",data[0][0][0])
# --- 3. MODES ---
def all_same(items):
    i = []
    for x in items:
        for y in x:
            for z in y:
                i.append( z == items[0][0][0] )
    return all( i )


if RUN_MODE == 'heatmap_anim':
    # ---------------------------------------------------------
    # 2D Heatmap Animation
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    print('data[?_1] = data[?_2]:', all_same(data) )
    heatmap = ax.imshow(
        data[0],
        cmap='coolwarm',
        # cmap='winter',
        interpolation='nearest',
    )
    
    ax.invert_yaxis()
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.tick_params(axis='both', which='major', labelsize=8)

    def update_heatmap(frame):
        heatmap.set_data(data[frame])
        # heatmap.set_clim(0, np.max(data))
        heatmap.set_clim(np.min(data), np.max(data))
        # heatmap.set_clim(0, 1)
        # heatmap.set_clim(0, 200000)
        # heatmap.set_clim(np.min(data[frame]), np.max(data[frame]))
        # heatmap.set_clim(0, np.max(data[frame]))
        # heatmap.set_clim(np.min(data[frame]), np.max(data[frame]))
        # ax.set_title(f"Simulation time {(frame*dt*STEP):.4f}, frame {frame}", fontsize=20)
        ax.set_title(f"frame {frame}", fontsize=20)
        return [heatmap]

    ani = FuncAnimation(fig, update_heatmap, frames=num_frames, interval=100)
    ani.save(ani_name + ".mp4", writer="ffmpeg", fps=48, dpi=200)
    plt.show()

elif RUN_MODE == 'line_anim':
    # ---------------------------------------------------------
    # 1D Line Plot Animation (Spatial Cut)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    x_data = np.arange(res)

    if TARGET_AXIS == 'row':
        y_data = data[0, TARGET_INDEX, :]
        title_str = f"Row {TARGET_INDEX}"
        global_min = np.min(data[:, TARGET_INDEX, :])
        global_max = np.max(data[:, TARGET_INDEX, :])
    else:
        y_data = data[0, :, TARGET_INDEX]
        title_str = f"Column {TARGET_INDEX}"
        global_min = np.min(data[:, :, TARGET_INDEX])
        global_max = np.max(data[:, :, TARGET_INDEX])

    line, = ax.plot(x_data, y_data, lw=2, color='blue')

    ax.set_xlim(0, res - 1)
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

    ani = FuncAnimation(fig, update_line, frames=num_frames, interval=50)
    
    save_name = f"1D_anim_{TARGET_AXIS}_{TARGET_INDEX}.mp4"
    print(f"Saving animation to {save_name}...")
    ani.save(save_name, writer="ffmpeg", fps=20, dpi=200)
    plt.show()

elif RUN_MODE == 'line_anim_phase_snap':
    # ---------------------------------------------------------
    # 1D Line Plot Animation, with phase snapshots
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    x_data = np.arange(res)

    if TARGET_AXIS == 'row':
        y_data = data[0, TARGET_INDEX, :]
        title_str = f"Row {TARGET_INDEX}"
        global_min = np.min(data[:, TARGET_INDEX, :])
        global_max = np.max(data[:, TARGET_INDEX, :])
    else:
        y_data = data[0, :, TARGET_INDEX]
        title_str = f"Column {TARGET_INDEX}"
        global_min = np.min(data[:, :, TARGET_INDEX])
        global_max = np.max(data[:, :, TARGET_INDEX])

    line, = ax.plot(x_data, y_data, lw=2, color='blue')

    ax.set_xlim(0, res - 1)
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

    ani = FuncAnimation(fig, update_line, frames=num_frames, interval=200)
    
    save_name = f"1D_anim_{TARGET_AXIS}_{TARGET_INDEX}.mp4"
    print(f"Saving animation to {save_name}...")
    ani.save(save_name, writer="ffmpeg", fps=20, dpi=200)
    plt.show()


elif RUN_MODE == 'line_static':
    # ---------------------------------------------------------
    # Static Multiple Lines (Spatial Cut at Selected Frames)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    x_data = np.arange(res)
    
    valid_frames = [f for f in FRAMES_TO_PLOT if f < num_frames]

    for frame in valid_frames:
        if TARGET_AXIS == 'row':
            y_data = data[frame, TARGET_INDEX, :]
            title_str = f"Row {TARGET_INDEX}"
        else:
            y_data = data[frame, :, TARGET_INDEX]
            title_str = f"Column {TARGET_INDEX}"
            
        ax.plot(x_data, y_data, lw=2, label=f"Frame {frame} (t={(frame*dt*STEP):.4f})")

    ax.set_xlim(0, res - 1)
    ax.set_xlabel("Cell Number", fontsize=FONT)
    ax.set_ylabel("Amplitude", fontsize=FONT)
    ax.set_title(f"1D Spatial Profiles Over Time ({title_str})", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=LABEL)
    
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

elif RUN_MODE == 'point_static':
    # ---------------------------------------------------------
    # NEW FEATURE 3: Temporal History for Specific Points
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Calculate the exact physical time for each frame
    # Time array = [0, 1*dt*STEP, 2*dt*STEP, ... ]
    # time_data = np.arange(num_frames) * dt * STEP
    time_data = np.arange(num_frames)
    
    # Loop through each requested point and extract its time history
    for point in POINTS_TO_PLOT:
        r, c = point
        
        # Ensure the requested point is within the simulation domain bounds
        if 0 <= r < res and 0 <= c < res:
            # Slicing syntax [:, r, c] grabs the value at (r, c) across all frames
            y_data = data[:, r, c]
            ax.plot(time_data, y_data, lw=2, label=f"Cell (row={r}, col={c})")
        else:
            print(f"Warning: Point {point} is out of bounds for a {res}x{res} grid.")

    # Formatting
    ax.set_xlim(0, time_data[-1])
    # ax.set_xlabel("Time (s)", fontsize=FONT)
    ax.set_xlabel("Frame (-)", fontsize=FONT)
    ax.set_ylabel("Amplitude", fontsize=FONT)
    ax.set_title("Time History of Selected Cells", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=LABEL)
    
    # Add legend and grid
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    # plt.savefig("point_history.png", dpi=200) # Uncomment to save
    plt.show()

else:
    print("Invalid RUN_MODE selected. Please check your configuration.")
