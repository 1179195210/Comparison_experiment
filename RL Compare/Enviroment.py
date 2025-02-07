import numpy as np
from Coordinate import Coordinate
class GridEnvironment:
    def __init__(self):
        self.state_size = (10, 10, 10)  # 10x10x10网格，每个位置4个通道
        self.grid_size = (10, 10, 10)
        self.action_space = [(x, y, z) for x in range(10) for y in range(10) for z in range(10)]
        self.state = [0] * 1000

        # 障碍物列表
        self.special_coordinates = [(1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 1, 2), (2, 1, 2), (2, 1, 3),
                                    (2, 1, 4), (2, 1, 5), (3, 2, 5), (3, 2, 6), (3, 2, 7), (4, 2, 7),
                                    (4, 3, 7), (4, 3, 8), (4, 3, 9), (4, 4, 7), (4, 4, 8), (4, 4, 9),
                                    (4, 5, 7), (5, 5, 8), (5, 5, 9),
                                    (6, 5, 3), (6, 5, 4), (6, 5, 7), (6, 6, 4), (6, 6, 5),
                                    (7, 1, 0), (7, 2, 0), (7, 2, 1), (7, 3, 1), (7, 3, 2),
                                    (7, 4, 2), (7, 4, 3), (7, 5, 3), (7, 6, 4), (7, 6, 5), (7, 6, 6),
                                    (8, 7, 2), (8, 7, 3), (8, 7, 4), (8, 8, 2), (9, 8, 1), (9, 8, 2),
                                    (9, 9, 0), (9, 9, 1)]

        # 初始化起点和终点坐标
        self.start_coordinate = None
        self.end_coordinate = None

    def reset(self, start, end):
        self.state = [0] * 1000
        coor = Coordinate()
        coor.set_start_by_coordinates(self.state, start[0], start[1], start[2])
        coor.set_end_by_coordinates(self.state, end[0], end[1], end[2])
        for coord in self.special_coordinates:
            index = coord[0]  + coord[1] * 10 + coord[2]* 100
            self.state[index] = 5

        return self.state
