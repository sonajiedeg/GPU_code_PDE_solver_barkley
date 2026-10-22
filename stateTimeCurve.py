import matplotlib.pyplot as plt
import numpy as np


def read_and_plot(filenameX, filenameY, qty):
    # Read ASCII text files directly into NumPy arrays
    t = np.loadtxt(filenameX)
    y = np.loadtxt(filenameY)

    print("----------->",t.size, y.size)

    plt.figure(figsize=(10, 5))
    plt.plot(t[:], y, label="sol.y", color="r", linestyle="-", linewidth=1.5)
    plt.xlabel("Time (sol.t)")
    plt.ylabel(qty)
    plt.title(qty)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


# read_and_plot("t_44000.00_df_200.00_s_0.txt", "x_44000.00_df_200.00_s_0.txt")
read_and_plot("t_25_df_50_Pa_75_R0_80sol.txt", "x_25_df_50_Pa_75_R0_80sol.txt", "x")
read_and_plot("t_25_df_50_Pa_75_R0_80sol.txt", "R_25_df_50_Pa_75_R0_80sol.txt", "R")
plt.show()


