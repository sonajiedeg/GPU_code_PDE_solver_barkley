import os
import matplotlib.pyplot as plt
import numpy as np


def calculate_transfer_function(filenameX, filenameY, freq, df_val, f_driver):
    """Calculates amplitude transfer and phase offset for time series."""
    t = np.fromfile(filenameX, dtype=np.float64)
    y = np.fromfile(filenameY, dtype=np.float64)

    # Reference Antinode position formula from stateTimeCurve logic
    c = 1500.0
    lam = c / f_driver
    dAx = 1623.7864077669904e-6
    Af = 2.0 * np.pi * dAx / lam
    ANPos = -Af * np.sin(2 * np.pi * df_val / f_driver * t) + 0.5

    omega = 2 * np.pi * freq

    # 1. Fit Input/Reference Signal (ANPos)
    A_mat1 = np.vstack([np.cos(omega * t), np.sin(omega * t), np.ones_like(t)]).T
    p1, _, _, _ = np.linalg.lstsq(A_mat1, ANPos, rcond=None)
    amp1 = np.sqrt(p1[0] ** 2 + p1[1] ** 2)
    phase1 = np.arctan2(-p1[1], p1[0])

    # 2. Fit Simulated Signal (y)
    A_mat2 = np.vstack([np.cos(omega * t), np.sin(omega * t), np.ones_like(t)]).T
    p2, _, _, _ = np.linalg.lstsq(A_mat2, y, rcond=None)
    amp2 = np.sqrt(p2[0] ** 2 + p2[1] ** 2)
    phase2 = np.arctan2(-p2[1], p2[0])

    # 3. Calculate Transfer Metrics
    amplitude_transfer = amp2 / amp1 if amp1 != 0 else np.nan
    phase_offset = phase2 - phase1

    # Wrap phase offset to [-pi, pi] and convert to degrees
    phase_offset = (phase_offset + np.pi) % (2 * np.pi) - np.pi
    phase_offset_deg = np.degrees(phase_offset)

    return {
        "amplitude_transfer": amplitude_transfer,
        "phase_offset_deg": phase_offset_deg,
    }


def generate_heatmaps():
    # ==========================================
    # 1. SPECIFY YOUR PARAMETER ARRAYS HERE
    # ==========================================
    df_values = [5]  # frequency offset values in Hz (e.g., 75)
    R0_values = [10**((k)/10)*10**(-5) - 0e-5 for k in range(8)]
    R0_values = [ r*1e6 for r in R0_values ]
    Pa_values = [66.0e3 - 8.0e3*k for k in range(8)]
    Pa_values = [ p*1e-3 for p in Pa_values ]

    # Fixed driving configuration
    f_driver = 44e3 # 25 kHz (match your simulation setting)
    A_val = f_driver / 1000.0

    # Loop over each frequency offset (df) value
    for df in df_values:
        print(f"Processing heatmaps for frequency offset df: {df} Hz...")

        nrows = len(Pa_values)
        ncols = len(R0_values)

        # Initialize matrices for grid storage (filled with NaN for missing data)
        gain_matrix = np.full((nrows, ncols), np.nan)
        phase_matrix = np.full((nrows, ncols), np.nan)

        # Iterate through the grid: Rows = Pa, Cols = R0
        for r_idx, Pa in enumerate(Pa_values):
            for c_idx, R0 in enumerate(R0_values):
                t_filename = f".data/5_25_125/t_{A_val:.0f}_df_{df:.0f}_Pa_{Pa:.0f}_R0_{R0:.0f}sol.bin"
                print(t_filename)
                x_filename = f".data/5_25_125/x_{A_val:.0f}_df_{df:.0f}_Pa_{Pa:.0f}_R0_{R0:.0f}sol.bin"

                if os.path.exists(t_filename) and os.path.exists(x_filename):
                    try:
                        # Frequency for fitting based on df offset relative to driver frequency
                        freq = df / f_driver
                        res = calculate_transfer_function(
                            t_filename, x_filename, freq, df, f_driver
                        )
                        gain_matrix[r_idx, c_idx] = res["amplitude_transfer"]
                        phase_matrix[r_idx, c_idx] = res["phase_offset_deg"]
                    except Exception as e:
                        print(
                            f"  [Error] Computing TF for Pa={Pa}, R0={R0}: {e}"
                        )
                else:
                    print(
                        f"  [Missing] Files not found for Pa={Pa}, R0={R0}"
                    )

        # Configure custom colormap where NaN values show up as Black
        cmap_gain = plt.cm.viridis.copy()
        cmap_gain.set_bad(color="black")

        cmap_phase = plt.cm.coolwarm.copy()  # Divergent colormap
        cmap_phase.set_bad(color="black")

        # ==========================================
        # 2. PLOT GAIN HEATMAP
        # ==========================================
        fig, ax = plt.subplots(figsize=(7, 6))
        im_gain = ax.imshow(
            gain_matrix, cmap=cmap_gain, aspect="equal", interpolation="nearest"
        )
        cbar_gain = fig.colorbar(im_gain, ax=ax)
        cbar_gain.set_label("Amplitude Gain ($A_2 / A_1$)", fontsize=10)

        ax.set_xticks(np.arange(ncols))
        ax.set_yticks(np.arange(nrows))
        ax.set_xticklabels([str(r) for r0, r in enumerate(R0_values) if True] # R0 labels
                           if False else [str(r) for r in R0_values])
        ax.set_xticklabels([f"{r}" for r in R0_values])
        ax.set_yticklabels([f"{p}" for p in Pa_values])
        
        ax.set_xlabel(r"Initial Bubble Size $R_0$ ($\mu$m)", fontsize=11)
        ax.set_ylabel(r"Driving Pressure $P_a$ (kPa)", fontsize=11)
        ax.set_title(f"Amplitude Gain Heatmap (df = {df} Hz)", fontsize=12)

        plt.tight_layout()
        gain_svg_name = f"gain_df_{df:.0f}.svg"
        plt.savefig(gain_svg_name, format="svg", dpi=300)
        print(f"Saved vector graphic: {gain_svg_name}")
        plt.close(fig)

        # ==========================================
        # 3. PLOT PHASE OFFSET HEATMAP
        # ==========================================
        fig, ax = plt.subplots(figsize=(7, 6))
        
        # Center the divergent colormap perfectly around 0 degrees (White = 0)
        vmax = np.nanmax(np.abs(phase_matrix))
        if np.isnan(vmax) or vmax == 0:
            vmax = 180.0 # Default bounds if array is empty

        im_phase = ax.imshow(
            phase_matrix,
            cmap=cmap_phase,
            vmin=-vmax,
            vmax=vmax,
            aspect="equal",
            interpolation="nearest",
        )
        cbar_phase = fig.colorbar(im_phase, ax=ax)
        cbar_phase.set_label("Phase Offset (Degrees)", fontsize=10)

        ax.set_xticks(np.arange(ncols))
        ax.set_yticks(np.arange(nrows))
        ax.set_xticklabels([f"{r}" for r in R0_values])
        ax.set_yticklabels([f"{p}" for p in Pa_values])

        ax.set_xlabel(r"Initial Bubble Size $R_0$ ($\mu$m)", fontsize=11)
        ax.set_ylabel(r"Driving Pressure $P_a$ (kPa)", fontsize=11)
        ax.set_title(f"Phase Offset Heatmap (df = {df} Hz, 0° = White)", fontsize=12)

        plt.tight_layout()
        phase_svg_name = f"phase_df_{df:.0f}.svg"
        plt.savefig(phase_svg_name, format="svg", dpi=300)
        print(f"Saved vector graphic: {phase_svg_name}")
        plt.close(fig)


if __name__ == "__main__":
    generate_heatmaps()
