import matplotlib.pyplot as plt
import numpy as np


def read_and_plot(filenameX, filenameY, qty):
    # Read ASCII text files directly into NumPy arrays
    # t = np.loadtxt(filenameX)
    # y = np.loadtxt(filenameY)
    t = np.fromfile(filenameX, dtype=np.float64)
    y = np.fromfile(filenameY, dtype=np.float64)


    # anti node pos (only based on the offset, not concidering the transient and bounary effects)
    if qty=="x":
        ff = 75
        f = 25e3
        c = 1500.0
        lam = c/f
        dAx = 1623.7864077669904e-6
        Af = 2.0 * np.pi * dAx / lam
        ANPos = np.zeros_like(t)
        ANPos = -Af * np.sin( 2*np.pi*ff/f*t ) + 0.5

    print("----------->",t.size, y.size)

    plt.figure(figsize=(10, 5))
    if qty=="R":
        plt.plot(t[:], y, label="R", color="r", linestyle="-", linewidth=1.5)
    if qty=="x":
        plt.plot(t[:], y, label="x, simulated", color="r", linestyle="-", linewidth=1.5)
        plt.plot(t[:], ANPos, label="x, set by phase offset", color="b", linestyle="-", linewidth=1.5)
    if True:
        plt.figure(figsize=(10,5))
        idx = [ i for i in range(len(t)) ]
        plt.plot(idx,t, label="t(k)", color="k", linestyle="-", linewidth=1.5)
    plt.xlabel("Time (sol.t)")
    plt.ylabel(qty)
    plt.title(qty)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


# read_and_plot("t_44000.00_df_200.00_s_0.txt", "x_44000.00_df_200.00_s_0.txt")
read_and_plot("t_25_df_75_Pa_75_R0_40sol.bin", "x_25_df_75_Pa_75_R0_40sol.bin", "x")
read_and_plot("t_25_df_75_Pa_75_R0_40sol.bin", "R_25_df_75_Pa_75_R0_40sol.bin", "R")
plt.show()


