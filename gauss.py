import numpy as np
import matplotlib.pyplot as plt

# sim resolution
res = 256
# pressure ic
p = np.zeros([res,res])
# square domain length
L = 1e-3 # m
# ic param.s
sigx = L/7 # m
sigy = L/5 # m
pa = 1e5 # Pa
# p0 = 101325 # Pa
p0 = 0 # Pa

def ic(x,y):
    return p0 + pa*np.exp(-1/2*( (x/sigx)**2 + (y/sigy)**2 ))

dx = L / res
xarr = np.linspace(-(L+dx)/2, (L+dx)/2, res)

for i, x in enumerate(xarr):
    for j, y in enumerate(xarr):
        p[i,j] = ic( x, y )

# im = plt.imshow(p, cmap='cool')
im = plt.imshow(p, cmap='coolwarm')
cbar = plt.colorbar(im, boundaries=np.linspace(0,200000,5))
plt.gca().invert_yaxis()
plt.show()

with open("p.txt","w") as f:
    for j in range( (res) ):
        for i in range(res):
            f.write(str(p[i,j]) + "\n")
