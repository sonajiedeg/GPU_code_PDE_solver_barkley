import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.ion()
# ----- material and bc parameters -----
rho = 1e3 # density
c = 1500 # speed of sound in water
f = 20e1 # transducer frequency
d = 10e-6 # transducer oscillation amplitude in [m]
omega = 2 * 3.14159265359 * f
# ----- simulation parameters -----
it = 5000 # iteration step number
res = 256 # number of pixels
size = res**2
dx = (c/f) * 3 / 256 # spatial discretization length
sigma = .01 # Courant number
dt = sigma * dx / c # time discretization, with CFL criterion
# transducer position:
T_min = 64 # is incuded
T_max = 192 # is icluded
print('dt from CFL: ', dt)
# TODO: examine effects of transducer frequency on max dt
# dt = min(dt, 1/20e4)
# print('dt: ',dt)
snapshot = 1 # if 1 take snapshot
snap_mod = 11 # take snapshot if k % snap_mod == 0

ua = np.zeros([res + 2,res + 2])
ub = np.zeros([res + 2,res + 2])
uc = np.zeros([res + 2,res + 2])
va = np.zeros([res,res])
vb = np.zeros([res,res])

# Create figure
fig, ax = plt.subplots(figsize=(9, 6))
ax.invert_yaxis()
ax.set_xlabel('j')
ax.set_ylabel('i')

# ----- list of points to be examined/plotted for examining the Transducer-Wall bc interracions
PlotPoints = [[256,63],[256,64],[256,65]]
uPlot = [[],[],[]]

# ----------1st step----------
k = 0
for i in range(res):
    for j in range(res):
        ua[i+1,j+1] = ub[i+1,j+1] + dt * vb[i,j]

        lap = 0
        lap += -20 * ub[i+1,j+1]
        lap +=   4 * ub[i+1,j+1+1]
        lap +=   4 * ub[i+1,j+1-1]
        lap +=   4 * ub[i+1+1,j+1]
        lap +=   4 * ub[i+1-1,j+1]
        lap +=   1 * ub[i+1-1,j+1-1]
        lap +=   1 * ub[i+1+1,j+1-1]
        lap +=   1 * ub[i+1+1,j+1+1]
        lap +=   1 * ub[i+1-1,j+1+1]

        va[i,j] = vb[i,j] + dt * c**2 / ( 6 * dx**2 ) * lap

# bc 
for j in range(res + 2):
    if 64 < j and j < 192: 
        # transducer
        ua[res+1,j] = ua[res,j] + dt * rho * d * omega * np.sin( omega * dt * (k) )
    else:
        # wall
        ua[res+1,j] = ua[res,j]

    # free
    ua[0,j] = 0
for i in range(res):
    # walls
    ua[i,0] = ua[i,1]
    ua[i,res+1] = ua[i,res]


for k in range(1, int(it), 2):
    # save prev field var
    uc = ub.copy()
    # iteration step
    for i in range(res):
        for j in range(res):
            ub[i+1,j+1] = 2 * ua[i+1,j+1] - uc[i+1,j+1] 
            lap = 0
            lap += -4* ua[i+1,j+1]
            lap +=   1 * ua[i+1,j+1+1]
            lap +=   1 * ua[i+1,j+1-1]
            lap +=   1 * ua[i+1+1,j+1]
            lap +=   1 * ua[i+1-1,j+1]
            # lap +=   1 * ua[i+1-1,j+1-1]
            # lap +=   1 * ua[i+1+1,j+1-1]
            # lap +=   1 * ua[i+1+1,j+1+1]
            # lap +=   1 * ua[i+1-1,j+1+1]
            ub[i+1,j+1] += dt**2 * c**2 / dx**2 * lap
    # bc 
    for j in range(res + 2):
        # if 64 < j and j < 192: 
        if T_min <= j <= T_max:
            # transducer
            ub[res+1,j] = ub[res,j] + dx * rho * d * omega**2 * np.sin( omega * dt * ( k ) ) - dx**2 / 2 / c**2 / dt**2 * ( ub[res,j] - 2 * ua[res,j] + uc[res,j] )
            # ub[res+1,j] = ub[res,j] + dt * rho * d * omega * np.sin( omega * dt * (k) )
        else:
            # wall
            ub[res+1,j] = ub[res,j] - dx**2 / 2 / c**2 / dt**2 * ( ub[res,j] - 2 * ua[res,j] + uc[res,j] )
        # free
        ub[0,j] = 0
    for i in range(res):
        # walls
        ub[i,0] = ub[i,1]  + dx**2 / 2 / c**2 / dt**2 * ( ub[i,1] - 2 * ua[i,1] + uc[i,1] )
        # ub[i,res+1] = ub[i,res]
        ub[i,res+1] = ub[i,res] - dx**2 / 2 / c**2 / dt**2 * ( ub[i,res] - 2 * ua[i,res] + uc[i,res] )
        ub[i,res+1] = ub[i,res] - dx**2 / 2 / c**2 / dt**2 * ( ub[i,res] - 2 * ua[i,res] + uc[i,res] )

    # ----- T-W bc debug:
    for z in range(len(PlotPoints)):
        uPlot[z].append( ub[ PlotPoints[z][0], PlotPoints[z][1] ] )

    # save prev field var
    uc = ua.copy()
    # iteration step
    for i in range(res):
        for j in range(res):
            ua[i+1,j+1] = 2 * ub[i+1,j+1] - uc[i+1,j+1] 
            lap = 0
            lap += -4* ub[i+1,j+1]
            lap +=   1 * ub[i+1,j+1+1]
            lap +=   1 * ub[i+1,j+1-1]
            lap +=   1 * ub[i+1+1,j+1]
            lap +=   1 * ub[i+1-1,j+1]
            # lap +=   1 * ub[i+1-1,j+1-1]
            # lap +=   1 * ub[i+1+1,j+1-1]
            # lap +=   1 * ub[i+1+1,j+1+1]
            # lap +=   1 * ub[i+1-1,j+1+1]
            ua[i+1,j+1] += dt**2 * c**2 / dx**2 * lap
    # bc 
    for j in range(res + 2):
        # if 64 < j and j < 192: 
        if T_min <= j <= T_max:
            # transducer
            ua[res+1,j] = ua[res,j] + dx * rho * d * omega**2 * np.sin( omega * dt * ( k ) ) - dx**2 / 2 / c**2 / dt**2 * ( ua[res,j] - 2 * ub[res,j] + uc[res,j] )
            # ua[res+1,j] = ua[res,j] + dt * rho * d * omega * np.sin( omega * dt * (k) )
        else:
            # wall
            ua[res+1,j] = ua[res,j] - dx**2 / 2 / c**2 / dt**2 * ( ua[res,j] - 2 * ub[res,j] + uc[res,j] )
        # free
        ua[0,j] = 0
    for i in range(res):
        # walls
        ua[i,0] = ua[i,1]  + dx**2 / 2 / c**2 / dt**2 * ( ua[i,1] - 2 * ub[i,1] + uc[i,1] )
        # ua[i,res+1] = ua[i,res]
        ua[i,res+1] = ua[i,res] - dx**2 / 2 / c**2 / dt**2 * ( ua[i,res] - 2 * ub[i,res] + uc[i,res] )

    # ----- T-W bc debug:
    for z in range(len(PlotPoints)):
        uPlot[z].append( ua[ PlotPoints[z][0], PlotPoints[z][1] ] )


    if snapshot and k % snap_mod == 0:
        heatmap = ax.imshow(
            ua,
            cmap='winter',
            interpolation='none',
            # vmin=vmin_value,
            # vmax=vmax_value
            )

        ax.set_title(f"k = {k}, max = {np.max(ua)}")
        ax.invert_yaxis()

        
        plt.draw()                # redraw
        plt.pause(0.001)          # allow GUI to update


# ----- T-W bc debug plot
n = range(len(uPlot[0])) 
print(len(n), len(uPlot[0]))
plt.figure(2)
for i in range(len(PlotPoints)):
    plt.plot(n,uPlot[i],label=PlotPoints[i])
plt.grid()
plt.legend()
plt.show()
plt.pause(10000)





