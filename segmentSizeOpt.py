import matplotlib.pyplot as plt
segmentSize = [1024*(k+1) for k in range(10)]
runtime = [44266, 44265]

plt.plot(segmentSize, runtime, color="r", linestyle="-")
plt.show()
