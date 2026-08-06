import h5py
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d


dir_path = "gauss_free/gauss_256/domain/"
search_pattern = os.path.join(dir_path, "domain_0.*.h5")
file_list = sorted(glob.glob(search_pattern))
num_steps = len(file_list)

# ================================================================================
# 1. GENERATE SPATIAL INDEX MAPPING FROM MESH TOPOLOGY
# ================================================================================
print("Building spatial mapping from mesh topology...")

with h5py.File(file_list[0], "r") as f:
    # Read vertex IDs and vertex coordinates
    cell_vertex_ids = f["mesh topology/cell vertex IDs"][:]          # Shape: (65536, 8)[cite: 1]
    vertex_coords = f["mesh topology/cell vertex coordinates"][:]      # Shape: (147968, 3)[cite: 1]

# Compute cell center (x, y) coordinates by averaging vertex positions
cell_x = vertex_coords[cell_vertex_ids, 0].mean(axis=1)
cell_y = vertex_coords[cell_vertex_ids, 1].mean(axis=1)

# Extract unique sorted 1D spatial grid coordinates
unique_x = np.unique(np.round(cell_x, decimals=9))
unique_y = np.unique(np.round(cell_y, decimals=9))

# Map floating-point cell centers to discrete integer grid indices [0..255]
ix = np.searchsorted(unique_x, np.round(cell_x, decimals=9))
iy = np.searchsorted(unique_y, np.round(cell_y, decimals=9))
'''
# ================================================================================
# 2. EXTRACT DATA USING SPATIAL MAPPING
# ================================================================================
pressure_data = np.zeros((num_steps, 256, 256), dtype=np.float64)
time_data = np.zeros(num_steps, dtype=np.float64)

print(f"Reading {num_steps} files using coordinate mapping...")

for i, file_path in enumerate(file_list):
    file_name = os.path.basename(file_path)
    sim_time = float(file_name.split('.')[1]) * 1e-9
    time_data[i] = sim_time
    
    with h5py.File(file_path, "r") as f:
        # Extract 1D pressure array (65536,)[cite: 1]
        raw_pressure = f['cell data/pressure'][:].flatten()
        
        # Re-assign data into data[n_timestep][m_x][k_y]
        pressure_data[i, ix, iy] = raw_pressure

print("Extraction complete. Grid successfully reconstructed!")
'''

 # ================================================================================
# 2. EXTRACT DATA USING SPATIAL MAPPING
# ================================================================================
pressure_data = np.zeros((num_steps, 256, 256), dtype=np.float64)
time_data = np.zeros(num_steps, dtype=np.float64)

print(f"Reading {num_steps} files using coordinate mapping...")

for i, file_path in enumerate(file_list):
    file_name = os.path.basename(file_path)
    sim_time = float(file_name.split('.')[1]) * 1e-9
    time_data[i] = sim_time
    
    with h5py.File(file_path, "r") as f:
        # Extract 1D pressure array (65536,)
        raw_pressure = f['cell data/pressure'][:].flatten()
        
        # Re-assign data into data[n_timestep][m_x][k_y]
        pressure_data[i, ix, iy] = raw_pressure

# --- LOOP ENDS HERE ---

# Apply the rotation exactly ONCE to the fully populated array
pressure_data = np.rot90(pressure_data, k=-1, axes=(1, 2))

print("Extraction and rotation complete!")

# ================================================================================
# ========== READ GPU SIMULATION DATA ============================================
# ================================================================================
res = 256
fname = "u_snap.bin"
p0 = 101325 # Pa

# Read the binary file into a NumPy array[cite: 2]
data = np.fromfile(fname, dtype=np.float32)
num_frames = data.size // (res * res)
data = data.reshape((num_frames, res, res))

print("--- GPU Data Loaded ---")
print("num_frames:", num_frames)
print("data size:", data.size)
print("expected per frame:", res*res)

# add ambient pressure[cite: 2]
data = data + p0  

# ================================================================================
# ========== COMPARE VAL AND GPU SIM DATA, WITH LINEAR INTERPOLATION =============
# ================================================================================

# 1. Create the time array for the GPU data
# Assuming total simulation time is identical to the validation data
total_sim_time = time_data[-1]
gpu_time = np.linspace(0, total_sim_time, num_frames)

# 2. Interpolate GPU data to match the adaptive validation timesteps
print("\nInterpolating GPU data to match HDF5 adaptive timesteps...")
# We use axis=0 to interpolate over the time dimension. 
# fill_value='extrapolate' catches edge cases from floating-point rounding.
interpolator = interp1d(gpu_time, data, axis=0, kind='linear', fill_value='extrapolate')
gpu_data_interp = interpolator(time_data)

# 3. Calculate the difference (GPU Solution - Validation Solution)
print("Calculating pressure differences...")
pressure_diff = gpu_data_interp - pressure_data


# ================================================================================
# ========== VISUAL COMPARISON & ANIMATION (SIDE-BY-SIDE) ========================
# ================================================================================

print("Generating side-by-side comparison animation...\n")

# Setup 1x3 subplots for GPU Solution, Validation Solution, and Difference
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

# 1. Determine common colormap limits for GPU & Validation so color intensities match
vmin_sol = min(gpu_data_interp.min(), pressure_data.min())*10**(-3)
vmax_sol = max(gpu_data_interp.max(), pressure_data.max())*10**(-3)

# 2. Determine symmetric colormap limits for the Difference panel
max_abs_diff = np.max(np.abs(pressure_diff))*10**(-3)
vmin_diff = -max_abs_diff
vmax_diff = max_abs_diff

# --- Subplot 1: GPU Simulation ---
cax1 = ax1.imshow(gpu_data_interp[0]*10**(-3), cmap='viridis', origin='lower', vmin=vmin_sol, vmax=vmax_sol)
ax1.set_title("GPU Simulation")
ax1.set_xlabel("X Grid")
ax1.set_ylabel("Y Grid")
fig.colorbar(cax1, ax=ax1, label="Pressure (kPa)", fraction=0.046, pad=0.04)

# --- Subplot 2: HDF5 Validation ---
cax2 = ax2.imshow(pressure_data[0]*10**(-3), cmap='viridis', origin='lower', vmin=vmin_sol, vmax=vmax_sol)
ax2.set_title("Validation, ALPACA")
ax2.set_xlabel("X Grid")
fig.colorbar(cax2, ax=ax2, label="Pressure (kPa)", fraction=0.046, pad=0.04)

# --- Subplot 3: Difference (GPU - Validation) ---
cax3 = ax3.imshow(pressure_diff[0]*10**(-3), cmap='RdBu_r', origin='lower', vmin=vmin_diff, vmax=vmax_diff)
ax3.set_title("Difference (GPU - Val)")
ax3.set_xlabel("X Grid")
fig.colorbar(cax3, ax=ax3, label="Δ Pressure (kPa)", fraction=0.046, pad=0.04)

# Main figure title for overall simulation time
fig_title = fig.suptitle(f'Simulation Time: {time_data[0]:.2e} s', fontsize=14, y=0.98)
plt.tight_layout()

# Update function for animation
def update(frame):
    cax1.set_data(gpu_data_interp[frame]*10**(-3))
    cax2.set_data(pressure_data[frame]*10**(-3))
    cax3.set_data(pressure_diff[frame]*10**(-3))
    fig_title.set_text(f'Simulation Time: {time_data[frame]:.2e} s')
    return cax1, cax2, cax3, fig_title

# Create animation object
ani = FuncAnimation(fig, update, frames=num_steps, blit=False, interval=100)

# To save as an MP4 or GIF, uncomment one of the following lines:
ani.save('side_by_side_comparison.mp4', writer='ffmpeg', fps=25)
# ani.save('side_by_side_comparison.gif', writer='pillow', fps=10)

plt.show()
