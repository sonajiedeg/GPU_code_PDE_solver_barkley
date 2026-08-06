import h5py
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ================================================================================
# ========== CONFIGURATION =======================================================
# ================================================================================
# HDF5 Path configuration
dir_path = "gauss_free/gauss_256/domain/"
search_pattern = os.path.join(dir_path, "domain_0.*.h5")
file_list = sorted(glob.glob(search_pattern))
num_steps = len(file_list)

# GPU configuration
res = 256
fname = "u_snap.bin"
p0 = 101325 # Pa

# Plotting configuration
LABEL = 12
FONT = 18
# List of (row, col) tuples you want to track over time
POINTS_TO_PLOT = [(128, 128), (128, 138), (64, 128)] 

# ================================================================================
# 1. READ VALIDATION DATA (HDF5) WITH SPATIAL MAPPING
# ================================================================================
print("Building spatial mapping from mesh topology...")

with h5py.File(file_list[0], "r") as f:
    cell_vertex_ids = f["mesh topology/cell vertex IDs"][:]
    vertex_coords = f["mesh topology/cell vertex coordinates"][:]

# Compute cell center (x, y) coordinates and map to discrete indices
cell_x = vertex_coords[cell_vertex_ids, 0].mean(axis=1)
cell_y = vertex_coords[cell_vertex_ids, 1].mean(axis=1)

unique_x = np.unique(np.round(cell_x, decimals=9))
unique_y = np.unique(np.round(cell_y, decimals=9))

ix = np.searchsorted(unique_x, np.round(cell_x, decimals=9))
iy = np.searchsorted(unique_y, np.round(cell_y, decimals=9))

# Extract the validation pressure data
pressure_data = np.zeros((num_steps, res, res), dtype=np.float64)
time_data = np.zeros(num_steps, dtype=np.float64)

print(f"Reading {num_steps} HDF5 files using coordinate mapping...")

for i, file_path in enumerate(file_list):
    file_name = os.path.basename(file_path)
    time_data[i] = float(file_name.split('.')[1]) * 1e-9
    
    with h5py.File(file_path, "r") as f:
        raw_pressure = f['cell data/pressure'][:].flatten()
        pressure_data[i, ix, iy] = raw_pressure

# Apply rotation ONCE to align the axes
pressure_data = np.rot90(pressure_data, k=-1, axes=(1, 2))

# ================================================================================
# 2. READ GPU SIMULATION DATA
# ================================================================================
print("Loading GPU data...")
data = np.fromfile(fname, dtype=np.float32)
num_frames = data.size // (res * res)
data = data.reshape((num_frames, res, res))
data = data + p0 

# ================================================================================
# 3. INTERPOLATE AND CALCULATE DIFFERENCE
# ================================================================================
total_sim_time = time_data[-1]
gpu_time = np.linspace(0, total_sim_time, num_frames)

print("Interpolating GPU data to match HDF5 adaptive timesteps...")
interpolator = interp1d(gpu_time, data, axis=0, kind='linear', fill_value='extrapolate')
gpu_data_interp = interpolator(time_data)

print("Calculating pressure differences...")
# Calculate the final difference array (GPU - Validation)
pressure_diff = gpu_data_interp - pressure_data

# ================================================================================
# 4. PLOT TIME HISTORY FOR SELECTED POINTS (POINT_STATIC)
# ================================================================================
print("Generating point-tracking plot...")

fig, ax = plt.subplots(figsize=(9, 6))

# Loop through each requested point and extract its time history
for point in POINTS_TO_PLOT:
    r, c = point
    
    # Ensure the requested point is within the simulation domain bounds
    if 0 <= r < res and 0 <= c < res:
        # Extract the exact (row, col) history across all interpolated timesteps
        y_data = pressure_diff[:, r, c]
        
        # Plot time vs pressure difference, scaling to kPa for readability
        ax.plot(time_data, y_data * 1e-3, lw=2, label=f"Cell (row={r}, col={c})")
    else:
        print(f"Warning: Point {point} is out of bounds for a {res}x{res} grid.")

# Formatting the Plot
ax.set_xlim(0, time_data[-1])
ax.set_xlabel("Time (s)", fontsize=FONT)
ax.set_ylabel("Δ Pressure (kPa)", fontsize=FONT)
ax.set_title("Pressure Difference Time History of Selected Cells", fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=LABEL)

# Add legend and grid
ax.legend(loc='best', fontsize=10)
ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
