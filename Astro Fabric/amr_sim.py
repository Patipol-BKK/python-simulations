import numpy as np
# Define Grid [level, (t_x, t_y), (l_x, l_y), [value_array], flag_status, id]
SIM_RESOLUTION = 16

def border_val(grids, focus_grid):
	t_x, t_y = focus_grid[1]
	l_x, l_y = focus_grid[2]
	level = focus_grid[0]
	top_arr = np.ones(l_x - t_x + 1, dtype=np.float64)
	btm_arr = np.ones(l_x - t_x + 1, dtype=np.float64)
	lft_arr = np.ones(l_y - t_y + 1, dtype=np.float64)
	rht_arr = np.ones(l_y - t_y + 1, dtype=np.float64)
	# print(top_arr.shape)
	for idx, i in enumerate(range(t_x, l_x + 1)):
		for grid in grids:
			if grid[5] != focus_grid[5]:
				if grid[0] > level:
					multiple = int(grid[0]/level)
					if int(np.floor((grid[2][1] + 1)/multiple)) >= t_x and \
					   int(np.floor(grid[1][0]/multiple)) <= i and \
					   int(np.floor(grid[2][0]/multiple)) >= i:
						top_arr_index = [(x, t_y * multiple - 1) for x in range(i*multiple,(i+1)*multiple)]
						top_arr_val = [grid[3][x[1]][x[0]] for x in top_arr_index]
						print(top_arr_val)

					if int(np.floor((grid[1][1] - 1)/multiple)) <= l_y and \
					   int(np.floor(grid[1][0]/multiple)) <= i and \
					   int(np.floor(grid[2][0]/multiple)) >= i:
						btm_arr_index = [(x, (l_y + 1) * multiple - 1) for x in range(i*multiple,(i+1)*multiple)]
						print(top_arr_index)

				else:
					multiple = int(level/grid[0])
					if int(np.floor((t_y - 1)/multiple)) <= grid[2][1] and \
					   grid[1][0]*multiple <= i and \
					   (grid[2][0] + 1) * multiple - 1 >= i:
						top_arr_index = (int(np.floor(i/multiple)), int(np.floor((t_y - 1)/multiple)))
						top_arr[idx] = grid[3][top_arr_index[1]][top_arr_index[0]]
						# print(top_arr_index)

					if int(np.floor((l_y + 1)/multiple)) >= grid[1][1] and \
					   grid[1][0]*multiple <= i and \
					   (grid[2][0] + 1) * multiple - 1 >= i:
						btm_arr_index = (int(np.floor(i/multiple)), int(np.floor((l_y - 1)/multiple)))
						btm_arr[idx] = grid[3][btm_arr_index[1]][btm_arr_index[0]]
						# print(btm_arr_index, l_y)
						
						# top_arr[idx] = grid[3][top_arr_index[1]][top_arr_index[0]]
	for idx, i in enumerate(range(t_y, l_y + 1)):
		for grid in grids:
			if grid[5] != focus_grid[5]:
				if grid[0] > level:
					multiple = int(grid[0]/level)
					if int(np.floor((grid[2][0] + 1)/multiple)) >= t_y and \
					   int(np.floor(grid[1][1]/multiple)) <= i and \
					   int(np.floor(grid[2][1]/multiple)) >= i:
						lft_arr_index = [(x, t_y * multiple - 1) for x in range(i*multiple,(i+1)*multiple)]

					if int(np.floor((grid[1][0] - 1)/multiple)) <= l_y and \
					   int(np.floor(grid[1][1]/multiple)) <= i and \
					   int(np.floor(grid[2][1]/multiple)) >= i:
						btm_arr_index = [(x, t_y*multiple - 1) for x in range(i*multiple,(i+1)*multiple)]
						btm_arr[idx] = grid[3][btm_arr_index[1]][btm_arr_index[0]]
						# print(btm_arr_index)

				else:
					multiple = int(level/grid[0])
					if int(np.floor((t_y - 1)/multiple)) <= grid[2][1] and \
					   grid[1][0]*multiple <= i and \
					   (grid[2][0] + 1) * multiple - 1 >= i:
						top_arr_index = (int(np.floor(i/multiple)), int(np.floor((t_y - 1)/multiple)))
						top_arr[idx] = grid[3][top_arr_index[1]][top_arr_index[0]]
						# print(top_arr_index)

					if int(np.floor((l_y + 1)/multiple)) >= grid[1][1] and \
					   grid[1][0]*multiple <= i and \
					   (grid[2][0] + 1) * multiple - 1 >= i:
						top_arr_index = (int(np.floor(i/multiple)), int(np.floor((l_y + 1)/multiple)))
						top_arr[idx] = grid[3][top_arr_index[1]][top_arr_index[0]]
						# print(top_arr_index)




grids = np.asarray([[SIM_RESOLUTION, (0,0), (SIM_RESOLUTION-1, SIM_RESOLUTION-1),
	np.zeros((SIM_RESOLUTION,SIM_RESOLUTION), dtype=np.float64), 
	np.zeros((SIM_RESOLUTION,SIM_RESOLUTION), dtype=np.int8), 0]])
sub_grid = np.asarray([SIM_RESOLUTION*2, (0,28), (31,29),
	np.zeros((2,SIM_RESOLUTION*2), dtype=np.float64),
	np.zeros((2,SIM_RESOLUTION*2), dtype=np.int8), 1])
sub_grid2 = np.asarray([SIM_RESOLUTION*4, (0,52), (63,55),
	np.zeros((4,SIM_RESOLUTION*4), dtype=np.float64),
	np.zeros((4,SIM_RESOLUTION*4), dtype=np.int8), 2])
grids = np.append(grids,np.asarray([sub_grid]),axis=0)
grids = np.append(grids,np.asarray([sub_grid2]),axis=0)

print(border_val(grids,sub_grid))

# print(grids)