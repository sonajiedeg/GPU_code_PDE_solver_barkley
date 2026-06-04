import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.ion()
# ----- material and bc parameters -----
rho = 1e3 # density
c = 1500 # speed of sound in water
f = 20e3 # transducer frequency
d = 10e-6 # transducer oscillation amplitude in [m]
omega = 2 * 3.14159265359 * f
# ----- simulation parameters -----
it = 50000 # iteration step number
res = 256 # number of pixels
size = res**2
dx = (c/f) * 3 / 256 # spatial discretization length
sigma = .01 # Courant number
dt = sigma * dx / c # time discretization, with CFL criterion
print('dt from CFL: ', dt)
# TODO: examine effects of transducer frequency on max dt
# dt = min(dt, 1/20e4)
# print('dt: ',dt)
snapshot = 1 # if 1 take snapshot
snap_mod = 9 # take snapshot if k % snap_mod == 0

# transducer position
tj1 = 64 # is transducer
tj2 = 192 # is transducer
# transducer - wall bc overlap with weights
nmbr = 0
w = ( np.linspace( 0, 1, num = nmbr+2 ) )[1:-1]
tau = 1e-3


ua = np.zeros([res + 2,res + 2])
ub = np.zeros([res + 2,res + 2])
uc = np.zeros([res + 2,res + 2])
va = np.zeros([res,res])
vb = np.zeros([res,res])

# Create figure
fig, ax = plt.subplots(figsize=(9, 6))
ax.invert_yaxis()
ax.set_xlabel('')
ax.set_ylabel('')

# ----------1st step----------
k = 0
for i in range(res):
    for j in range(res):
        uc[i+1,j+1] = ua[i+1,j+1] + dt * vb[i,j]

        lap = 0
        lap += -20 * ua[i+1,j+1]
        lap +=   4 * ua[i+1,j+1+1]
        lap +=   4 * ua[i+1,j+1-1]
        lap +=   4 * ua[i+1+1,j+1]
        lap +=   4 * ua[i+1-1,j+1]
        lap +=   1 * ua[i+1-1,j+1-1]
        lap +=   1 * ua[i+1+1,j+1-1]
        lap +=   1 * ua[i+1+1,j+1+1]
        lap +=   1 * ua[i+1-1,j+1+1]

        va[i,j] = vb[i,j] + dt * c**2 / ( 6 * dx**2 ) * lap

w_counter = 0
# bc 
for j in range(res + 2):
    if tj1 <= j <= tj1 :
        # transducer
        uc[res+1,j] = uc[res,j] + dt * rho * d * omega * np.sin( omega * dt * (k + 0) ) * ( 1 - np.exp( -(k + 0) * dt / tau ) )
    elif tj1 - nmbr <= j < tj1 and False:
        bc1 = uc[res,j] + dx * rho * d * omega**2 * np.sin( omega * dt * (k + 0) ) * ( 1 - np.exp( -(k + 0) * dt / tau ) ) - dx**2 / 2 / c**2 / dt**2 * ( uc[res,j] - 2 * ua[res,j] + ub[res,j] )
        bc2 = uc[res,j] - dx**2 / 2 / c**2 / dt**2 * ( uc[res,j] - 2 * ua[res,j] + ub[res,j] )
        uc[res+1,j] = w[w_counter] * bc1 + (1 - w[w_counter]) * bc2
        w_counter += 1
        if j == tj1-1:
            w_counter = 0
    elif tj2 < j <= tj2 + nmbr and False:
        bc1 = uc[res,j] - dx**2 / 2 / c**2 / dt**2 * ( uc[res,j] - 2 * ua[res,j] + ub[res,j] )
        bc2 = uc[res,j] + dx * rho * d * omega**2 * np.sin( omega * dt * ( k + 0 ) ) * ( 1 - np.exp( -(k + 0) * dt / tau ) ) - dx**2 / 2 / c**2 / dt**2 * ( uc[res,j] - 2 * ua[res,j] + ub[res,j] )
        uc[res+1,j] = w[w_counter] * bc1 + (1 - w[w_counter]) * bc2
        w_counter += 1
        if j == tj2+nmbr:
            w_counter = 0
    else:
        # wall
        uc[res+1,j] = uc[res,j]

    # free
    uc[0,j] = 0
for i in range(res):
    # walls
    uc[i,0] = uc[i,1]
    uc[i,res+1] = uc[i,res]

w_counter = 0
for k in range(1, int(it), 3):
    # iteration step { n+1: ub, n: uc, n-1: ua, n-2: ub }
    for i in range(res):
        for j in range(res):
            ub[i+1,j+1] = uc[i+1,j+1] + ua[i+1,j+1] - ub[i+1,j+1]
            lap = 0
            lap += -20 * uc[i+1,  j+1  ]
            lap +=   4 * uc[i+1,  j+1+1]
            lap +=   4 * uc[i+1,  j+1-1]
            lap +=   4 * uc[i+1+1,j+1  ]
            lap +=   4 * uc[i+1-1,j+1  ]
            lap +=   1 * uc[i+1-1,j+1-1]
            lap +=   1 * uc[i+1+1,j+1-1]
            lap +=   1 * uc[i+1+1,j+1+1]
            lap +=   1 * uc[i+1-1,j+1+1]
            ub[i+1,j+1] += 2 * dt**2 * c**2 / dx**2 / 6 * lap
    # bc 
    for j in range(res + 2):
        if tj1 <= j <= tj1 :
        # if 0 <= j <= res + 1:
            # transduaer
            ub[res+1,j] = ub[res,j] + dx * rho * d * omega**2 * np.sin( omega * dt * ( k + 1 ) ) * ( 1 - np.exp( -(k + 1) * dt / tau ) ) - dx**2 / 2 / c**2 / dt**2 * ( ub[res,j] - 2 * uc[res,j] + ua[res,j] )
            # ub[res+1,j] = ub[res,j] + dt * rho * d * omega * np.sin( omega * dt * (k) )
        elif tj1 - nmbr <= j < tj1 and False:
            bc1 = ub[res,j] + dx * rho * d * omega**2 * np.sin( omega * dt * ( k + 1 ) ) * ( 1 - np.exp( -(k + 1) * dt / tau ) ) - dx**2 / 2 / c**2 / dt**2 * ( ub[res,j] - 2 * uc[res,j] + ua[res,j] )
            bc2 = ub[res,j] - dx**2 / 2 / c**2 / dt**2 * ( ub[res,j] - 2 * uc[res,j] + ua[res,j] )
            ub[res+1,j] = w[w_counter] * bc1 + (1 - w[w_counter]) * bc2
            w_counter += 1
            if j == tj1-1:
                w_counter = 0
        elif tj2 < j <= tj2 + nmbr and False:
            bc1 = ub[res,j] - dx**2 / 2 / c**2 / dt**2 * ( ub[res,j] - 2 * uc[res,j] + ua[res,j] )
            bc2 = ub[res,j] + dx * rho * d * omega**2 * np.sin( omega * dt * ( k + 1 ) ) * ( 1 - np.exp( -(k + 1) * dt / tau ) ) - dx**2 / 2 / c**2 / dt**2 * ( ub[res,j] - 2 * uc[res,j] + ua[res,j] )
            ub[res+1,j] = w[w_counter] * bc1 + (1 - w[w_counter]) * bc2
            w_counter += 1
            if j == tj2+nmbr:
                w_counter = 0
        else:
            # wall
            ub[res+1,j] = ub[res,j] - dx**2 / 2 / c**2 / dt**2 * ( ub[res,j] - 2 * uc[res,j] + ua[res,j] )
        # free
        ub[0,j] = 0
    for i in range(res):
        # walls
        ub[i,0] = ub[i,1]  + dx**2 / 2 / c**2 / dt**2 * ( ub[i,1] - 2 * uc[i,1] + ua[i,1] )
        # ub[i,res+1] = ub[i,res]
        ub[i,res+1] = ub[i,res] - dx**2 / 2 / c**2 / dt**2 * ( ub[i,res] - 2 * uc[i,res] + ua[i,res] )

    # iteration step { n+1: ua, n: ub, n-1: uc, n-2: ua }
    for i in range(res):
        for j in range(res):
            ua[i+1,j+1] = ub[i+1,j+1] + uc[i+1,j+1] - ua[i+1,j+1]
            lap = 0
            lap += -20 * ub[i+1,  j+1  ]
            lap +=   4 * ub[i+1,  j+1+1]
            lap +=   4 * ub[i+1,  j+1-1]
            lap +=   4 * ub[i+1+1,j+1  ]
            lap +=   4 * ub[i+1-1,j+1  ]
            lap +=   1 * ub[i+1-1,j+1-1]
            lap +=   1 * ub[i+1+1,j+1-1]
            lap +=   1 * ub[i+1+1,j+1+1]
            lap +=   1 * ub[i+1-1,j+1+1]
            ua[i+1,j+1] += 2 * dt**2 * c**2 / dx**2 / 6 * lap
    # bc 
    for j in range(res + 2):
        if tj1 <= j <= tj1 :
        # if 0 <= j <= res + 1:
            # transducer
            ua[res+1,j] = ua[res,j] + dx * rho * d * omega**2 * np.sin( omega * dt * ( k + 2 ) ) * ( 1 - np.exp( -(k + 2) * dt / tau ) ) - dx**2 / 2 / c**2 / dt**2 * ( ua[res,j] - 2 * ub[res,j] + uc[res,j] )
            # ua[res+1,j] = ua[res,j] + dt * rho * d * omega * np.sin( omega * dt * (k) )
        elif tj1 - nmbr <= j < tj1 and False:
            bc1 = ua[res,j] + dx * rho * d * omega**2 * np.sin( omega * dt * ( k + 2 ) ) * ( 1 - np.exp( -(k + 3) * dt / tau ) ) - dx**2 / 2 / c**2 / dt**2 * ( ua[res,j] - 2 * ub[res,j] + uc[res,j] )
            bc2 = ua[res,j] - dx**2 / 2 / c**2 / dt**2 * ( ua[res,j] - 2 * ub[res,j] + uc[res,j] )
            ua[res+1,j] = w[w_counter] * bc1 + (1 - w[w_counter]) * bc2
            w_counter += 1
            if j == tj1-1:
                w_counter = 0
        elif tj2 < j <= tj2 + nmbr and False:
            bc1 = ua[res,j] - dx**2 / 2 / c**2 / dt**2 * ( ua[res,j] - 2 * ub[res,j] + uc[res,j] )
            bc2 = ua[res,j] + dx * rho * d * omega**2 * np.sin( omega * dt * ( k + 2 ) ) * ( 1 - np.exp( -(k + 2) * dt / tau ) ) - dx**2 / 2 / c**2 / dt**2 * ( ua[res,j] - 2 * ub[res,j] + uc[res,j] )
            ua[res+1,j] = w[w_counter] * bc1 + (1 - w[w_counter]) * bc2
            w_counter += 1
            if j == tj2+nmbr:
                w_counter = 0
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



    # iteration step { n+1: uc, n: ua, n-1: ub, n-2: uc }
    for i in range(res):
        for j in range(res):
            uc[i+1,j+1] = ua[i+1,j+1] + ub[i+1,j+1] - uc[i+1,j+1]
            lap = 0
            lap += -20 * ua[i+1,  j+1  ]
            lap +=   4 * ua[i+1,  j+1+1]
            lap +=   4 * ua[i+1,  j+1-1]
            lap +=   4 * ua[i+1+1,j+1  ]
            lap +=   4 * ua[i+1-1,j+1  ]
            lap +=   1 * ua[i+1-1,j+1-1]
            lap +=   1 * ua[i+1+1,j+1-1]
            lap +=   1 * ua[i+1+1,j+1+1]
            lap +=   1 * ua[i+1-1,j+1+1]
            uc[i+1,j+1] += 2 * dt**2 * c**2 / dx**2 / 6 * lap
    # bc 
    for j in range(res + 2):
        if tj1 <= j <= tj1 :
        # if 0 <= j <= res + 1:
            # transduber
            uc[res+1,j] = uc[res,j] + dx * rho * d * omega**2 * np.sin( omega * dt * ( k + 0 ) ) - dx**2 / 2 / c**2 / dt**2 * ( uc[res,j] - 2 * ua[res,j] + ub[res,j] )
            # uc[res+1,j] = uc[res,j] + dt * rho * d * omega * np.sin( omega * dt * (k) )
        elif tj1 - nmbr <= j < tj1 and False:
            bc1 = uc[res,j] + dx * rho * d * omega**2 * np.sin( omega * dt * ( k + 0 ) ) - dx**2 / 2 / c**2 / dt**2 * ( uc[res,j] - 2 * ua[res,j] + ub[res,j] )
            bc2 = uc[res,j] - dx**2 / 2 / c**2 / dt**2 * ( uc[res,j] - 2 * ua[res,j] + ub[res,j] )
            uc[res+1,j] = w[w_counter] * bc1 + (1 - w[w_counter]) * bc2
            w_counter += 1
            if j == tj1-1:
                w_counter = 0
        elif tj2 < j <= tj2 + nmbr and False:
            bc1 = uc[res,j] - dx**2 / 2 / c**2 / dt**2 * ( uc[res,j] - 2 * ua[res,j] + ub[res,j] )
            bc2 = uc[res,j] + dx * rho * d * omega**2 * np.sin( omega * dt * ( k + 0 ) ) - dx**2 / 2 / c**2 / dt**2 * ( uc[res,j] - 2 * ua[res,j] + ub[res,j] )
            uc[res+1,j] = w[w_counter] * bc1 + (1 - w[w_counter]) * bc2
            w_counter += 1
            if j == tj2+nmbr:
                w_counter = 0

        else:
            # wall
            uc[res+1,j] = uc[res,j] - dx**2 / 2 / c**2 / dt**2 * ( uc[res,j] - 2 * ua[res,j] + ub[res,j] )
        # free
        uc[0,j] = 0
    for i in range(res):
        # walls
        uc[i,0] = uc[i,1]  + dx**2 / 2 / c**2 / dt**2 * ( uc[i,1] - 2 * ua[i,1] + ub[i,1] )
        # uc[i,res+1] = uc[i,res]
        uc[i,res+1] = uc[i,res] - dx**2 / 2 / c**2 / dt**2 * ( uc[i,res] - 2 * ua[i,res] + ub[i,res] )

    if snapshot and ( k - 1 ) % snap_mod == 0:
        
        heatmap = ax.imshow(
            uc,
            cmap='winter',
            interpolation='none',
            # vmin=vmin_value,
            # vmax=vmax_value
            )

        ax.set_title(f"k = {k}, max = {np.max(ua)}")
        
        plt.draw()                # redraw
        plt.pause(0.001)          # allow GUI to update
