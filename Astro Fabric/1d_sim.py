import numpy as np
import matplotlib.pyplot as plt
import threading

# Declare discrete constants
LENGTH = 100
NODES = 10000								# Also includes end points
THREAD_COUNT = 1
chunk = NODES/THREAD_COUNT

delta_x = LENGTH/(NODES - 1)	# Step size
delta_t = 0.1								# Time step size

# Define velocity array
vel_array 	= np.zeros((3, NODES), dtype=np.float32)		# (2, NODES)-size array for keeping tract of prev node velocities
order		= np.asarray([0,1,2])

prev_arrays	= np.zeros((50, NODES), dtype=np.float32)
prev_order 	= np.asarray([i for i in range(50)])

def update_order(order):
	for i in range(order.shape[0]):
		order[i] = (order[i] + order.shape[0] - 1) % order.shape[0]

def simulate_next(vel, dx, dt, c, order):
	shape_size = vel.shape[1]

	vel[order[0]][1:shape_size-1] = 2 * vel[order[1]][1:shape_size-1] - vel[order[2]][1:shape_size-1] \
				+ (c * delta_t / delta_x)**2 \
				* (vel[order[1]][2:shape_size] - 2*vel[order[1]][1:shape_size-1] + vel[order[1]][0:shape_size-2])

# Set dx/dt = 0
vel_array[order[0]][1:vel_array.shape[1] - 1] = vel_array[order[1]][1:vel_array.shape[1]-1] \
												- 0.5*(5*delta_t/delta_x)**2*(vel_array[order[1]][2:vel_array.shape[1]] \
												- 2*vel_array[order[1]][1:vel_array.shape[1]-1] + vel_array[order[1]][0:vel_array.shape[1]-2])
update_order(order)

count = 0
co = 0

accum = np.zeros(NODES, dtype=np.float32)
while True:
	vel_array[order[1]][0] = np.sin(co/500)

	simulate_next(vel_array, delta_x, delta_t, 0.1, order)

	prev_arrays[prev_order[0]] = vel_array[order[0]]
	update_order(prev_order)

	prev_sd = prev_arrays.var(axis=0)
	print(prev_sd)
	# accum = (accum*co+vel_array[order[0]])/(co+1)
	if count >= 100:
		count = 0
		plt.plot(prev_sd)
		# plt.plot(vel_array[order[0]])
		# plt.ylim(-2, 2)
		# plt.ylim(0, 0.00008)
		plt.pause(0.0001)
		plt.clf()

	count += 1
	co += 1
	update_order(order)
plt.show()
print(vel_array)