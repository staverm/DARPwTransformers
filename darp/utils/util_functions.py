import torch
import drawSvg as draw
import matplotlib.pyplot as plt
import numpy as np

def indice_map2image(indice_map, image_size):
    # x = Id // image_size ; y = Id%image_size
    return np.reshape(indice_map, (image_size, image_size))


def indices2image(indice_list, image_size):
    indice_map = torch.zeros(image_size**2)
    for i, indice in enumerate(indice_list):
        indice_map[indice.item()] = 1.
        if i==(len(indice_list) - 1):
            indice_map[indice.item()] = 0.5
    return indice_map2image(indice_map, image_size)


def instance2world(indice_list, type_vector, image_size):
    indice_map = np.zeros(image_size**2)
    for i, indice in enumerate(indice_list):
        indice_map[indice] = type_vector[i]
    return indice_map2image(indice_map, image_size)


def image_coordonates2indices(coord, image_size):
    x, y = coord
    return x*image_size + y


def indice2image_coordonates(indice, image_size):
    # x = Id // image_size ; y = Id%image_size
    return indice // image_size, indice%image_size


def coord2int(coord):
    number_precision = 3
    new_coord = int(round(coord, number_precision)*(10**number_precision))
    return new_coord


def distance(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2))


def float_equality(f1, f2, eps=0.001):
    return abs(f1 - f2) < eps


def GAP_function(cost, best_cost):
    if best_cost is None :
        return None
    plus = cost - best_cost
    return 100 * plus / best_cost

def time2int(time):
    number_precision = 0
    new_coord = int(round(time, number_precision)*(10**number_precision))
    return new_coord

