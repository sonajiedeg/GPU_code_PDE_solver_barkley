import subprocess
import itertools

# 1. Define your parameter arrays/lists
# (Example: 3 parameters, each with multiple values to test)

# param1_vals = [6.870229007633588, 9.16030534351145, 14.50381679389313, 19.84732824427481, 24.427480916030536, 29.00763358778626, 34.35114503816794, 38.93129770992366, 48.85496183206107, 59.541984732824424, 68.70229007633588, 78.62595419847328, 89.31297709923665, 99.23664122137404, 119.08396946564885, 148.85496183206106, 198.4732824427481]


param1_vals = [5, 25, 125]              # df
param2_vals = [10.0e3 + 8.0e3*k for k in range(1,4)]           # Pa
# param2_vals = [42.0e3 + 8.0e3*k for k in range(4)]           # Pa
param3_vals = [10**((k)/10)*10**(-5) - 0e-5 for k in range(8)]      # R0

# If you want a full grid sweep (all possible combinations):
# itertools.product generates every combination of your parameters
parameter_combinations = list(itertools.product(param1_vals, param2_vals, param3_vals))


# Alternatively, if your parameters are paired step-by-step (zipped lists):
# parameter_combinations = list(zip(param1_vals, param2_vals, param3_vals))

# Slice the list to start from the second simulation (index 1)
# parameter_combinations[1:] drops the first element (index 0)
remaining_combinations = parameter_combinations[1:]

print(f"Starting parameter sweep. Total runs: {len(parameter_combinations)}\n")

# 2. Loop through each set of parameters
for i, (p1, p2, p3) in enumerate(remaining_combinations, start=2):
    print(f"--- Run {i}/{len(parameter_combinations)}: p1={p1}, p2={p2}, p3={p3} ---")
    
    # Construct the command as a list of strings
    # Convert all parameters to strings so subprocess can pass them as command-line arguments
    cmd = ["./kernel.exe", str(p1), str(p2), str(p3), str(8192), str(0)]
    
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
