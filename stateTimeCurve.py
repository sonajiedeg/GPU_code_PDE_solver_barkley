import matplotlib.pyplot as plt
import numpy as np


def read_and_plot(filenameX, filenameY):
    # Read ASCII text files directly into NumPy arrays
    t = np.loadtxt(filenameX)
    y = np.loadtxt(filenameY)

    print("----------->",t.size, y.size)

    plt.figure(figsize=(10, 5))
    plt.plot(t, y, label="sol.y", color="b", linestyle="--", marker="o", linewidth=1.5)
    plt.xlabel("Time (sol.t)")
    plt.ylabel("State (sol.y)")
    plt.title("Solution Plot")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


read_and_plot("t_44000.00_df_200.00_s_0.txt", "x_44000.00_df_200.00_s_0.txt")



'''
import numpy as np
import matplotlib.pyplot as plt

def read_and_plot(filenameX,filenameY):
    with open(filenameX, "r") as f:
        size = np.fromfile(f, dtype=np.uint64, count=1)[0]
        t = np.fromfile(f, dtype=np.float32, count=size)
    with open(filenameY, "r") as f:
        size = np.fromfile(f, dtype=np.uint64, count=1)[0]
        y = np.fromfile(f, dtype=np.float32, count=size)

    # 3. Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(t, y, label="sol.y", color="b", linewidth=1.5)
    plt.xlabel("Time (sol.t)")
    plt.ylabel("State (sol.y)")
    plt.title("Solution Plot")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Run post-processing
read_and_plot("t_44000.00_df_200.00_s_0.txt","x_44000.00_df_200.00_s_0.txt")
'''
