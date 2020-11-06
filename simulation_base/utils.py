import numpy as np

def get_indecies(shape, x_points, y_points):
    x_indecies = np.round(np.linspace(0, shape[0]-1, x_points)).astype(int)
    y_indecies = np.round(np.linspace(0, shape[1]-1, y_points)).astype(int)

    indecies = list()
    for x in x_indecies:
        for y in y_indecies:
            indecies.append([x, y])
    indecies = np.array(indecies)

    return indecies