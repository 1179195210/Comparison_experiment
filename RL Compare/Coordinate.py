import numpy as np
import math

class Coordinate:
    def __init__(self):
        self.coorinates = [(x, y, z) for x in range(10) for y in range(10) for z in range(10)]

        self.features = [0] * len(self.coorinates)

        self.OBSTACLE = -1
        self.START = 2000
        self.END = 3000


    def get_index_by_coordinates(self, x, y, z):

        return x + y*10 + z*100

    def index_to_coordinates(self, index):
        if 0 <= index < 1000:
            z = index // 100
            index %= 100
            y = index // 10
            x = index % 10
            return x, y, z
        else:
            raise IndexError("Index out of bounds.")

    def set_feature_by_coordinates(self, state, x, y, z, value):
        """
        Set the feature value for a given 3D coordinates.
        """
        index = self.get_index_by_coordinates(x, y, z)
        if 0 <= index < len(self.features):
            state[index] = value
            return state
        else:
            raise ValueError("Coordinates out of bounds.")

    def set_start_by_coordinates(self, state, x, y, z):
         return self.set_feature_by_coordinates(state, x, y, z, self.START)

    def set_end_by_coordinates(self, state, x, y, z):
        return self.set_feature_by_coordinates(state, x, y, z, self.END)
