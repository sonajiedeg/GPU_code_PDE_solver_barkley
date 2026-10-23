import numpy as np

def calculate_transfer_function(filenameX, filenameY, freq):
    """
    Calculates amplitude transfer and phase offset for non-uniform time series.
    
    Parameters:
    - t1, x1: Time array and signal array for input 1
    - t2, x2: Time array and signal array for input 2
    - freq: Known matching frequency of the signals (in Hz)
    """

    # t = np.loadtxt(filenameX)
    # y = np.loadtxt(filenameY)

    t = np.fromfile(filenameX, dtype=np.float64)
    y = np.fromfile(filenameY, dtype=np.float64)
    ff = 5 # the frequency offset, later to be added as an input
    f = 44e3
    c = 1500.0
    lam = c/f
    dAx = 1623.7864077669904e-6
    Af = 2.0 * np.pi * dAx / lam
    ANPos = np.zeros_like(t)
    ANPos = -Af * np.sin( 2*np.pi*ff/f*t ) + 0.5


    omega = 2 * np.pi * freq
    
    # 1. Fit Signal 1: x1(t) = a1*cos(omega*t) + b1*sin(omega*t) + c1
    A_mat1 = np.vstack([np.cos(omega * t), np.sin(omega * t), np.ones_like(t)]).T
    p1, _, _, _ = np.linalg.lstsq(A_mat1, ANPos, rcond=None)
    amp1 = np.sqrt(p1[0]**2 + p1[1]**2)
    phase1 = np.arctan2(-p1[1], p1[0]) # in radians
    
    # 2. Fit Signal 2: ANPos(t) = a2*cos(omega*t) + b2*sin(omega*t) + c2
    A_mat2 = np.vstack([np.cos(omega * t), np.sin(omega * t), np.ones_like(t)]).T
    p2, _, _, _ = np.linalg.lstsq(A_mat2, y, rcond=None)
    amp2 = np.sqrt(p2[0]**2 + p2[1]**2)
    phase2 = np.arctan2(-p2[1], p2[0]) # in radians
    
    # 3. Calculate Transfer Metrics
    amplitude_transfer = amp2 / amp1  # Gain ratio (Signal 2 / Signal 1)
    phase_offset = phase2 - phase1    # Phase difference in radians
    
    # Wrap phase offset to [-pi, pi]
    phase_offset = (phase_offset + np.pi) % (2 * np.pi) - np.pi
    
    return {
        "amplitude_transfer": amplitude_transfer,
        "phase_offset_rad": phase_offset,
        "phase_offset_deg": np.degrees(phase_offset),
        "amp1": amp1,
        "amp2": amp2
    }

# --- Example Usage ---
# Generate dummy non-uniform time data
freq = 5.0 / 44000.0 # 2 Hz

result = calculate_transfer_function(".data/5_25_125/t_44_df_5_Pa_42_R0_50sol.bin", ".data/5_25_125/x_44_df_5_Pa_42_R0_50sol.bin", freq)
print(f"Amplitude Transfer (Gain): {result['amplitude_transfer']:.4f}")
print(f"Phase Offset: {result['phase_offset_deg']:.2f} degrees")
