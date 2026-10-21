import numpy as np
import matplotlib.pyplot as plt

# convert the pixel coordinates of the A(df) graph to (df,A) points
pSW = (664,237) # (0,0)
pNE = (926,31) # (200,1500)
pdata = [(673,14),(676,51),(683,126),(690,154),
         (696,168),(702,181),(709,184),(715,192),
         (728,205),(742,208),(754,212),(767,216),
         (781,218),(794,221),(820,225),(859,225),
         (924,228)]
pxLx = pNE[0] - pSW[0]
pxLy = pNE[1] - pSW[1]
dfL = 200
AL = 1500
ddata = [[0]*2 for _ in range(len(pdata))]
print(ddata)
for idx, px in enumerate(pdata):
    dfi = dfL * ( pdata[idx][0] - pSW[0] ) / pxLx
    Ai = AL * ( pdata[idx][1] - pSW[1] ) / pxLy
    ddata[idx][0] = dfi
    ddata[idx][1] = Ai
    print(f'px {idx}: {pdata[idx]} [px], ({dfi}[Hz],{Ai}[um])')

plt.figure(1)
# plt.plot(ddata[:][0],ddata[:][1])
x = [ xi[0] for xi in ddata ]
y = [ xi[1] for xi in ddata ]
plt.plot(x,y)
plt.show()
print(ddata[:][0])
print(ddata[1][0])


x = [xi[0] for xi in ddata]
print("Frequencies:", x)
