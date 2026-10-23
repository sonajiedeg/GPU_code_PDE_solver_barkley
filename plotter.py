import os
import matplotlib.pyplot as plt
import numpy as np


def generate_parameter_sweep_plots():
    # ==========================================
    # 1. SPECIFY YOUR PARAMETER ARRAYS HERE
    # ==========================================
    df_values = [5]  # frequency offset values in Hz (e.g., 75)
    R0_values = [10**((k)/10)*10**(-5) - 0e-5 for k in range(8)]
    R0_values = [ r*1e6 for r in R0_values ]
    Pa_values = [66.0e3 - 8.0e3*k for k in range(8)]
    Pa_values = [ p*1e-3 for p in Pa_values ]

    # Fixed driving parameters based on your C++ setup
    f_driver = 44e3  # 25 kHz
    c = 1500.0
    lam = c / f_driver
    dAx = 1623.7864077669904e-6
    Af = 2.0 * np.pi * dAx / lam

    # Maximum points to plot per curve to prevent OOM/RAM exhaustion in WSL
    MAX_PLOT_POINTS = 2000

    # Loop over each frequency offset (df) value
    for df in df_values:
        for qty in ["R", "x"]:
            print(
                f"Processing grid for quantity: {qty}, frequency offset df: {df} Hz..."
            )

            nrows = len(Pa_values)
            ncols = len(R0_values)

            # Create a figure with square subplots arrangement
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(ncols * 2.5, nrows * 2.5),
                subplot_kw=dict(box_aspect=1),
                sharex=False,
                sharey=False,
            )

            # Ensure axes is always a 2D numpy array
            if nrows == 1 and ncols == 1:
                axes = np.array([[axes]])
            elif nrows == 1:
                axes = axes[np.newaxis, :]
            elif ncols == 1:
                axes = axes[:, np.newaxis]

            # Iterate through the grid: Rows = Pa, Cols = R0
            for r_idx, Pa in enumerate(Pa_values):
                for c_idx, R0 in enumerate(R0_values):
                    ax = axes[r_idx, c_idx]

                    # Pointing to the .data/ folder as confirmed
                    A_val = f_driver / 1000.0
                    t_filename = f".data/5_25_125/t_{A_val:.0f}_df_{df:.0f}_Pa_{Pa:.0f}_R0_{R0:.0f}sol.bin"
                    print(t_filename)
                    data_filename = f".data/5_25_125/{qty}_{A_val:.0f}_df_{df:.0f}_Pa_{Pa:.0f}_R0_{R0:.0f}sol.bin"

                    # Check if both time and data files exist
                    if os.path.exists(t_filename) and os.path.exists(
                        data_filename
                    ):
                        try:
                            t = np.fromfile(t_filename, dtype=np.float64)
                            y = np.fromfile(data_filename, dtype=np.float64)

                            # Downsample large arrays to prevent memory overflow
                            if len(t) > MAX_PLOT_POINTS:
                                step = len(t) // MAX_PLOT_POINTS
                                t_plot = t[::step]
                                y_plot = y[::step]
                            else:
                                t_plot = t
                                y_plot = y

                            if qty == "R":
                                ax.plot(
                                    t_plot,
                                    y_plot,
                                    color="r",
                                    linestyle="-",
                                    linewidth=0.8,
                                )
                            elif qty == "x":
                                # Calculate theoretical antinode position based on offset df
                                ANPos = (
                                    -Af
                                    * np.sin(
                                        2 * np.pi * df / f_driver * t_plot
                                    )
                                    + 0.5
                                )
                                ax.plot(
                                    t_plot,
                                    y_plot,
                                    label="Simulated",
                                    color="r",
                                    linestyle="-",
                                    linewidth=0.8,
                                )
                                ax.plot(
                                    t_plot,
                                    ANPos,
                                    label="Antinode",
                                    color="b",
                                    linestyle="--",
                                    linewidth=0.8,
                                )

                        except Exception as e:
                            print(
                                f"  [Error] Reading files for Pa={Pa}, R0={R0}: {e}"
                            )
                            ax.text(
                                0.5,
                                0.5,
                                "Read Error",
                                transform=ax.transAxes,
                                ha="center",
                                va="center",
                                fontsize=8,
                                color="orange",
                            )
                    else:
                        # File not found: keep plot blank to preserve grid structure
                        ax.text(
                            0.5,
                            0.5,
                            "Missing",
                            transform=ax.transAxes,
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="gray",
                        )
                        ax.set_xticks([])
                        ax.set_yticks([])

                    # Annotate outer labels for clarity
                    if r_idx == nrows - 1:
                        ax.set_xlabel(f"$R_0$={R0}$\\mu m$", fontsize=8)
                    else:
                        ax.tick_params(labelbottom=False)

                    if c_idx == 0:
                        ax.set_ylabel(f"$P_a$={Pa}kPa", fontsize=8)
                    else:
                        ax.tick_params(labelleft=False)

                    ax.tick_params(labelsize=6)

            # Overall titles and layout adjustment
            fig.suptitle(
                f"Parameter Sweep Grid: {qty}(t) | Driving Freq A={f_driver/1000.0} kHz, df={df} Hz",
                fontsize=12,
            )
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            # Save as vector graphic (.svg)
            output_svg_name = f"{qty}_{f_driver/1000.0:.0f}_df_{df:.0f}.svg"
            plt.savefig(output_svg_name, format="svg", dpi=300)
            print(f"Successfully saved vector graphic: {output_svg_name}")
            
            # Explicitly close figure to free memory back to WSL
            plt.close(fig)


if __name__ == "__main__":
    generate_parameter_sweep_plots()
