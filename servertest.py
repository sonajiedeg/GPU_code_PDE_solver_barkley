import subprocess
import itertools

# 1. Define your parameter arrays/lists
# (Example: 3 parameters, each with multiple values to test)

# param1_vals = [6.870229007633588, 9.16030534351145, 14.50381679389313, 19.84732824427481, 24.427480916030536, 29.00763358778626, 34.35114503816794, 38.93129770992366, 48.85496183206107, 59.541984732824424, 68.70229007633588, 78.62595419847328, 89.31297709923665, 99.23664122137404, 119.08396946564885, 148.85496183206106, 198.4732824427481]
param1_vals = [ 0.8, 1.3, 1.7, 2.5, 4 ]
# ================================================================================
# ========== CHECKLIST ===========================================================
# ================================================================================
'''
    0) is the gpu free to use?
    1) is df correct?
    2) is the Pa array matches the batch?
    3) is the R0 array matches the batch?
    4) is the R0 and antiNode correct 
            (suB (below) res R0 => antiNode = 1) 
            (suP (above) res R0 => antiNode = 0) 
    5) max step per segment mabye is irrelevant
    6) is the deviceID (DID) mathces the batch?
    7) if there was a restart, is the slicing of the parameters correct?
    8) if there was a restart, is the start in enum. correct?
'''
# sw0
'''
param3_vals = [59.0e-6]
param2_vals = [34.0e3]
antiNode = 1
DID = 0
'''
# sw1
'''
param3_vals = [70.0e-6]
param2_vals = [10.0e3]
antiNode = 1
DID = 0
'''
# sw2
'''
param3_vals = [83.0e-6]
param2_vals = [34.0e3]
antiNode = 0
DID = 1
'''
# sw3
param3_vals = [83.0e-6]
param2_vals = [34.0e3]
antiNode = 0
DID = 1


# tmux mapping? [0,1,2,3]
#                         -> R0[0,1,0,1]
#                         -> AN[1,0,1,0]
#                         -> Pa[0,0,1,1]
#                         -> Di[0,0,1,1]

# df
# param1_vals = [25]


# R0 0                                                                     <-----
# param3_vals = [10**((k)/10)*10**(-5) + 2e-5 for k in range(8)]
# R0 1                                                                     <-----
 # param3_vals = [10**((k)/10)*10**(-5) + 2e-5 for k in range(8,16)]

# antinode                                                                 <-----
# antiNode = 1

# Pa 0                                                                     <-----
# param2_vals = [10.0e3 + 8.0e3*k for k in range(8)]
# Pa 1                                                                     <-----
# param2_vals = [10.0e3 + 8.0e3*k for k in range(8,16)]

# device ID                                                                <-----
# DID = 1

# maxStepPerSegment
maxStepPerSegment = 8192

# If you want a full grid sweep (all possible combinations):
# itertools.product generates every combination of your parameters
parameter_combinations = list(itertools.product(param1_vals, param2_vals, param3_vals))


# Alternatively, if your parameters are paired step-by-step (zipped lists):
# parameter_combinations = list(zip(param1_vals, param2_vals, param3_vals))

# Slice the list to start from the second simulation (index 1)
# parameter_combinations[1:] drops the first element (index 0)
remaining_combinations = parameter_combinations
# remaining_combinations = parameter_combinations[1:]

print(f"Starting parameter sweep. Total runs: {len(parameter_combinations)}\n")

# 2. Loop through each set of parameters
for i, (p1, p2, p3) in enumerate(remaining_combinations, start=1):
    print(f"--- Run {i}/{len(parameter_combinations)}: p1={p1}, p2={p2}, p3={p3} ---")
    
    # Construct the command as a list of strings
    # Convert all parameters to strings so subprocess can pass them as command-line arguments
    cmd = ["./kernel.exe", str(p1), str(p2), str(p3), str(maxStepPerSegment), str(DID), str(antiNode)]
    
    try:
        # subprocess.run blocks execution here until kernel.exe completely finishes
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        
        # Optional: Print the output from your C++ executable if it prints to console
        if result.stdout:
            print("C++ Output:\n", result.stdout)
            
        print(f"Run {i} finished successfully.\n")
        
    except subprocess.CalledProcessError as e:
        print(f"Error: Run {i} failed with exit code {e.returncode}")
        if e.stderr:
            print("Error details:\n", e.stderr)
        break # Stop the sweep if a run fails (optional)

print("All parameter sweeps are complete!")
