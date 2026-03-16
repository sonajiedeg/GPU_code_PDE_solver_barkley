import numpy as np
import matplotlib.pyplot as plt

res = 256

dir = r"/mnt/c/Users/Gedei Janos Jonatan/Dokumentumok/LaTeX Projects/MSc-repo/3/SonoChem/GPU_code_PDE_solver_barkley/"
s1 = r"u_comp_test_5000.txt"
s1 = r"u_sol_uni.txt"


# Read data from files
with open(dir + s1, 'r') as file1:
    u_sol = np.array([float(file1.readline().strip()) for _ in range(res**2)])

# Reshape the arrays to 2D
u_sol = u_sol.reshape((res, res))

# Create the heatmap

# vmin_value = -0.5
# vmax_value = 0.5

plt.figure(figsize=(9, 6))
heatmap = plt.imshow(u_sol, cmap='winter', interpolation='nearest')
# heatmap = plt.imshow(u_sol, cmap='winter', interpolation='nearest', vmin=vmin_value, vmax=vmax_value)
plt.gca().invert_yaxis()
#plt.colorbar()
plt.axis('on')  # Turn off the grid

plt.xlabel('', fontsize=30, fontweight='bold')
plt.ylabel('', fontsize=30, fontweight='bold')

colorbar = plt.colorbar(heatmap)
# Increase the tick label font size for both axes
plt.tick_params(axis='both', which='major', labelsize=30)
colorbar.ax.tick_params(labelsize=30)

plt.show()

# print("%20f",u_sol.max())

print("{:.200f}".format(u_sol.max()))
