import numpy as np
import matplotlib.pyplot as plt
import threading
import time

# Declare discrete constants
LENGTH = 1000
NODES = 1001								# Also includes end points
THREAD_COUNT = 4
chunk = NODES/THREAD_COUNT

delta_x = 1									# Step size
delta_t = 0.1								# Time step size

# Define position array
pos 		= np.zeros((3, NODES, NODES), dtype=np.float64)		# (3, NODES)-size array for keeping tract of prev node velocities
fixed		= np.zeros((NODES, NODES), dtype=np.float64)
order		= np.asarray([0,1,2])

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


def simulate_next(pos, fixed, dx, dt, c, order):
	shape_size 	= pos.shape[1]
	c_const 	= c * delta_t / delta_x

	pos[order[0]][1:shape_size-1, 1:shape_size-1] = \
		- pos[order[2]][1:shape_size-1, 1:shape_size-1] + 2 * pos[order[1]][1:shape_size-1, 1:shape_size-1] \
		+ c_const**2 * (pos[order[1]][2:shape_size, 1:shape_size-1]  \
		- 2 * pos[order[1]][1:shape_size-1, 1:shape_size-1] + pos[order[1]][0:shape_size-2, 1:shape_size-1]) \
		+ c_const**2 * (pos[order[1]][1:shape_size-1, 2:shape_size]  \
		- 2 * pos[order[1]][1:shape_size-1, 1:shape_size-1] + pos[order[1]][1:shape_size-1, 0:shape_size-2])

	pos[order[0]][1:shape_size-1, 1:shape_size-1] *= np.abs(fixed[1:shape_size-1, 1:shape_size-1] - 1)

def simulate_range(pos, fixed, dx, dt, c, order, r1, r2):
	shape_size 	= pos.shape[1]
	c_const 	= c * delta_t / delta_x

	pos[order[0]][r1:r2, 1:shape_size-1] = \
		- pos[order[2]][r1:r2, 1:shape_size-1] + 2 * pos[order[1]][r1:r2, 1:shape_size-1] \
		+ c_const**2 * (pos[order[1]][r1+1:r2+1, 1:shape_size-1]  \
		- 2 * pos[order[1]][r1:r2, 1:shape_size-1] + pos[order[1]][r1-1:r2-1, 1:shape_size-1]) \
		+ c_const**2 * (pos[order[1]][r1:r2, 2:shape_size]  \
		- 2 * pos[order[1]][r1:r2, 1:shape_size-1] + pos[order[1]][r1:r2, 0:shape_size-2])

	pos[order[0]][r1:r2, 1:shape_size-1] *= np.abs(fixed[r1:r2, 1:shape_size-1] - 1)

def simulate_init(pos, fixed, dx, dt, c, order):
	shape_size 	= pos.shape[1]
	c_const 	= c * delta_t / delta_x

	pos[order[0]][1:shape_size-1, 1:shape_size-1] = pos[order[1]][1:shape_size-1, 1:shape_size-1] \
		+ 0.5 * c_const**2 * (pos[order[1]][2:shape_size, 1:shape_size-1]  \
		- 2 * pos[order[1]][1:shape_size-1, 1:shape_size-1] + pos[order[1]][0:shape_size-2, 1:shape_size-1]) \
		+ 0.5 * c_const**2 * (pos[order[1]][1:shape_size-1, 2:shape_size]  \
		- 2 * pos[order[1]][1:shape_size-1, 1:shape_size-1] + pos[order[1]][1:shape_size-1, 0:shape_size-2])

count = 0

fixed = get_pixalated_circle(NODES)
# fixed[int((NODES-1)/2)][302+int((NODES-1)/2)] = 1
# fixed[int((NODES-1)/2)][191+int((NODES-1)/2)] = 1
# fixed[int((NODES-1)/2)][96+int((NODES-1)/2)] = 1
# fixed[int((NODES-1)/2)][52+int((NODES-1)/2)] = 1
fixed[int((NODES-1)/2)][15+int((NODES-1)/2)] = 1
fixed[int((NODES-1)/2)][10+int((NODES-1)/2)] = 1
fixed[int((NODES-1)/2)][7+int((NODES-1)/2)] = 1
fixed[int((NODES-1)/2)][4+int((NODES-1)/2)] = 1

simulate_init(pos, fixed, delta_x, delta_t, 1, order)
order = update_order(order)
start = time.time()

val_max = 0
val_min = 0

est_mean = np.zeros((NODES, NODES), dtype=np.float64)
est_var = np.zeros((NODES, NODES), dtype=np.float64)

while True:
	pos[order[1]][int((NODES-1)/2)][int((NODES-1)/2)] = np.sin(count/10)
	# print(np.sin(count/1000))
	threads = list()
	for i in range(THREAD_COUNT):
		x = threading.Thread(target=simulate_range, args=(pos, fixed, delta_x, delta_t, 1, order, int(max(1,i*chunk)), int(min(NODES-1, (i+1)*chunk))))
		threads.append(x)
		x.start()

	for thread in threads:
		thread.join()
	# simulate_next(pos, fixed, delta_x, delta_t, 1.1, order)

	if count%10 == 0:
		val_max = max(np.amax(pos[order[0]]), val_max)
		val_min = min(np.amin(pos[order[0]]), val_min)
		plt.imshow(pos[order[0]]+fixed, vmin=val_min, vmax=val_max)
		plt.pause(0.0001)
		plt.clf()
	if count % 100 == 0 : 
		print(count, time.time()-start)
		start = time.time()

	count += 1
	order = update_order(order)

plt.show()
