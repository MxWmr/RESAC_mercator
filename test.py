import numpy as np
import matplotlib.pyplot as plt

arr = np.random.rand(20,4)

plt.figure(1)
plt.plot(arr,label=['1','2','3','4'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()