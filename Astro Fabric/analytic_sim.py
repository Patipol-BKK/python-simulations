import numpy as np
from fractions import Fraction, gcd
from functools import reduce
from scipy import special

import matplotlib.pyplot as plt

RESOLUTION = 801
R = (RESOLUTION-1)/2
A = 1
c = 299792458
l = 1

def lcm(a, b):
    return a * b // np.gcd(a, b)

def common_integer(*numbers):
    fractions = [Fraction(n).limit_denominator() for n in numbers]
    multiple  = reduce(lcm, [f.denominator for f in fractions])
    ints      = [f * multiple for f in fractions]
    divisor   = reduce(np.gcd, ints)
    return [int(n / divisor) for n in ints]

img = np.zeros((RESOLUTION,RESOLUTION),dtype=np.float64)
thetas = np.zeros((RESOLUTION,RESOLUTION),dtype=np.float64)
rs = np.zeros((RESOLUTION,RESOLUTION),dtype=np.float64)

for i in range(RESOLUTION):
	for j in range(RESOLUTION):
		rs[i][j] = np.sqrt((R - j)**2 + (R - i)**2)
		thetas[i][j] = np.arctan2(R - i, j- R)
		# img[i][j] = theta

img = A*np.sin(5*thetas)*special.jv(5,rs/(2*np.pi*l))
plt.imshow(img)
plt.show()
print(common_integer(0.4)[0]/0.4)
print(np.arctan2(1,6))