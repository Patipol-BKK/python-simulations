import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import threading
import time

from numba import vectorize
	
# Declare discrete constants
NODES = 501								# Also includes end points
THREAD_COUNT = 4
chunk = NODES/THREAD_COUNT
chunks = np.asarray([[int(max(1,i*chunk)), int(min(NODES-1, (i+1)*chunk))] for i in range(THREAD_COUNT)])
# print(chunks)

GRID_SIZE = 100
TIME_STEP = 0.2

# 0.02 AU/SEC
# 0.001 AU/SEC
# 1 AU
# 10 times

delta_x = 1									# Step size
delta_t = 0.1								# Time step size

# Define position array
pos 		= np.zeros((3, NODES, NODES), dtype=np.float64)		# (3, NODES)-size array for keeping tract of prev node velocities
fixed		= np.zeros((NODES, NODES), dtype=np.float64)
order		= np.asarray([0,1,2])

# Define initial planet condition [distance, orbital period, starting position]
planets 	= np.asarray([[0.387, 7600530.24, np.pi/2],[0.723, 19414162.944, -np.pi/2],[1, 31558149.504, 0],[1.523, 59354294.4, np.pi]])

def get_pixalated_circle(diameter):
	cir_array = np.zeros((diameter, diameter), dtype=np.int8)
	center = float(diameter) / 2
	for top_r in range(int(np.ceil(center))):
		height = np.sqrt(center**2 - (top_r - center)**2)
		cir_array[int(np.floor(center))-int(height)][top_r] = 1
		cir_array[int(np.floor(center))+int(height)][top_r] = 1
		cir_array[int(np.floor(center))-int(height)][int(np.floor(center))*2 - top_r] = 1
		cir_array[int(np.floor(center))+int(height)][int(np.floor(center))*2 - top_r] = 1

		cir_array[int(top_r)][int(np.floor(center))-int(height)] = 1
		cir_array[int(top_r)][int(np.floor(center))+int(height)] = 1
		cir_array[int(np.floor(center))*2 - top_r][int(np.floor(center))-int(height)] = 1
		cir_array[int(np.floor(center))*2 - top_r][int(np.floor(center))+int(height)] = 1

	# print(cir_array)
	return cir_array

def update_order(order):
	 return (order[0:order.shape[0]] + order.shape[0] - 1) % order.shape[0]

@vectorize(['float64[:](float64[:,:,:], float64[:,:], float64, float64, float64, int8[:])'], target='cuda')
def simulate_next(pos, fixed, dx, dt, c, order):
	shape_size 	= pos.shape[1]
	c_const 	= c * delta_t / delta_x

	for i in range(1, shape_size-1):
		for j in range(1, shape_size-1):
			pos[order[0]][i][j] = \
				- pos[order[2]][i][j] + 2 * pos[order[1]][i][j] \
				+ c_const**2 * (pos[order[1]][i+1][j]  \
				- 2 * pos[order[1]][i][j] + pos[order[1]][i-1][j]) \


	pos[order[0]][1:shape_size-1, 1:shape_size-1] = \
		- pos[order[2]][1:shape_size-1, 1:shape_size-1] + 2 * pos[order[1]][1:shape_size-1, 1:shape_size-1] \
		+ c_const**2 * (pos[order[1]][2:shape_size, 1:shape_size-1]  \
		- 2 * pos[order[1]][1:shape_size-1, 1:shape_size-1] + pos[order[1]][0:shape_size-2, 1:shape_size-1]) \
		+ c_const**2 * (pos[order[1]][1:shape_size-1, 2:shape_size]  \
		- 2 * pos[order[1]][1:shape_size-1, 1:shape_size-1] + pos[order[1]][1:shape_size-1, 0:shape_size-2])

# def simulate_next(pos, fixed, dx, dt, c, order):
# 	shape_size 	= pos.shape[1]
# 	c_const 	= c * delta_t / delta_x

# 	pos[order[0]][1:shape_size-1, 1:shape_size-1] = \
# 		- pos[order[2]][1:shape_size-1, 1:shape_size-1] + 2 * pos[order[1]][1:shape_size-1, 1:shape_size-1] \
# 		+ c_const**2 * (pos[order[1]][2:shape_size, 1:shape_size-1]  \
# 		- 2 * pos[order[1]][1:shape_size-1, 1:shape_size-1] + pos[order[1]][0:shape_size-2, 1:shape_size-1]) \
# 		+ c_const**2 * (pos[order[1]][1:shape_size-1, 2:shape_size]  \
# 		- 2 * pos[order[1]][1:shape_size-1, 1:shape_size-1] + pos[order[1]][1:shape_size-1, 0:shape_size-2])

# 	pos[order[0]][1:shape_size-1, 1:shape_size-1] *= np.abs(fixed[1:shape_size-1, 1:shape_size-1] - 1)

# def simulate_range(pos, fixed, dx, dt, c, order, r1, r2):
# 	shape_size 	= pos.shape[1]
# 	c_const 	= c * delta_t / delta_x

# 	pos[order[0]][r1:r2, 1:shape_size-1] = \
# 		- pos[order[2]][r1:r2, 1:shape_size-1] + 2 * pos[order[1]][r1:r2, 1:shape_size-1] \
# 		+ c_const**2 * (pos[order[1]][r1+1:r2+1, 1:shape_size-1]  \
# 		- 2 * pos[order[1]][r1:r2, 1:shape_size-1] + pos[order[1]][r1-1:r2-1, 1:shape_size-1]) \
# 		+ c_const**2 * (pos[order[1]][r1:r2, 2:shape_size]  \
# 		- 2 * pos[order[1]][r1:r2, 1:shape_size-1] + pos[order[1]][r1:r2, 0:shape_size-2])

# 	pos[order[0]][r1:r2, 1:shape_size-1] *= np.abs(fixed[r1:r2, 1:shape_size-1] - 1)

# def simulate_init(pos, fixed, dx, dt, c, order):
# 	shape_size 	= pos.shape[1]
# 	c_const 	= c * delta_t / delta_x

# 	pos[order[0]][1:shape_size-1, 1:shape_size-1] = pos[order[1]][1:shape_size-1, 1:shape_size-1] \
# 		+ 0.5 * c_const**2 * (pos[order[1]][2:shape_size, 1:shape_size-1]  \
# 		- 2 * pos[order[1]][1:shape_size-1, 1:shape_size-1] + pos[order[1]][0:shape_size-2, 1:shape_size-1]) \
# 		+ 0.5 * c_const**2 * (pos[order[1]][1:shape_size-1, 2:shape_size]  \
# 		- 2 * pos[order[1]][1:shape_size-1, 1:shape_size-1] + pos[order[1]][1:shape_size-1, 0:shape_size-2])

# def planet_update(planets, fixed):
# 	for planet in planets:
# 		center = float(NODES)/2
# 		fixed[int(center - GRID_SIZE * planet[0] * np.cos(planet[2]))][int(center - GRID_SIZE * planet[0] * np.sin(planet[2]))] = 0

# 		planet[2] += 2*np.pi/planet[1]*10*TIME_STEP
# 		# print(planet[2])
# 		fixed[int(center - GRID_SIZE * planet[0] * np.cos(planet[2]))][int(center - GRID_SIZE * planet[0] * np.sin(planet[2]))] = 1

# count = 1

# fixed = get_pixalated_circle(NODES)
# # fixed[int((NODES-1)/2)][302+int((NODES-1)/2)] = 1
# # fixed[int((NODES-1)/2)][191+int((NODES-1)/2)] = 1
# # fixed[int((NODES-1)/2)][96+int((NODES-1)/2)] = 1
# # fixed[int((NODES-1)/2)][52+int((NODES-1)/2)] = 1
# # fixed[int((NODES-1)/2)][75+int((NODES-1)/2)] = 1
# # fixed[int((NODES-1)/2)][50+int((NODES-1)/2)] = 1
# # fixed[int((NODES-1)/2)][35+int((NODES-1)/2)] = 1
# # fixed[int((NODES-1)/2)][20+int((NODES-1)/2)] = 1

# simulate_init(pos, fixed, delta_x, delta_t, 1, order)
# order = update_order(order)
# start = time.time()

# val_max = 0
# val_min = 0

# est_mean = np.zeros((2, NODES, NODES), dtype=np.float64)
# mean_order = np.asarray([0,1])
# est_var = np.zeros((NODES, NODES), dtype=np.float64)
# while True:
# 	planet_update(planets, fixed)
# 	pos[order[1]][int((NODES-1)/2)][int((NODES-1)/2)] = np.sin(count/200)
# 	# print(np.sin(count/1000))
# 	threads = list()
# 	for i in range(THREAD_COUNT):
# 		x = threading.Thread(target=simulate_range, args=(pos, fixed, delta_x, delta_t, 1, order, int(max(1,i*chunk)), int(min(NODES-1, (i+1)*chunk))))
# 		threads.append(x)
# 		x.start()

# 	for thread in threads:
# 		thread.join()

# 	# simulate_next(pos, fixed, delta_x, delta_t, 1.1, order)

# 	if count%20 == 0:
# 		val_max = max(np.amax(pos[order[0]]), val_max)
# 		val_min = min(np.amin(pos[order[0]]), val_min)
# 		plt.imshow((pos[order[0]]+fixed), vmin=val_min, vmax=val_max)
# 		plt.pause(0.01)
# 		plt.clf()
# 	if count % 200 == 0 : 
# 		print(count, time.time()-start)
# 		start = time.time()

# 	# est_mean[mean_order[0]] = est_mean[mean_order[1]] + (pos[order[0]] - est_mean[mean_order[1]])/float(count)
# 	# est_var 				= est_var + ((pos[order[0]]-est_mean[mean_order[1]])*(pos[order[0]]-est_mean[mean_order[0]])-est_var)/float(count)

# 	order = update_order(order)
# 	mean_order = update_order(mean_order)
# 	count += 1

# plt.show()
