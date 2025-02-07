import numpy as np
import math

class Coordinate:
    def __init__(self):
        # Generate the 3D coordinates for the 1000 points
        self.coordinates = [(x, y, z) for x in range(10) for y in range(10) for z in range(10)]

        self.features = [0] * len(self.coordinates)  # 0 represents unvisited

        self.OBSTACLE = -1  # Obstacle
        self.UNVISITED = 0  # Unvisited
        self.VISITED = 1  # Visited
        self.START = 2     # Start point
        self.END = 3       # End point

    def get_index_by_coordinates(self, x, y, z):
        """
        Convert 3D coordinates to the index in the features list.
        """
        return x + y * 10 + z * 100

    def set_feature_by_coordinates(self, x, y, z, value):
        """
        Set the feature value for a given 3D coordinates.
        """
        index = self.get_index_by_coordinates(x, y, z)
        if 0 <= index < len(self.features):
            self.features[index] = value
        else:
            raise ValueError("Coordinates out of bounds.")

    def set_obstacle_by_coordinates(self, x, y, z):
        """
        Set the feature value to obstacle at the given 3D coordinates.
        """
        self.set_feature_by_coordinates(x, y, z, self.OBSTACLE)

    def set_start_by_coordinates(self, x, y, z):
        """
        Set the feature value to start point at the given 3D coordinates.
        """
        self.set_feature_by_coordinates(x, y, z, self.START)

    def set_end_by_coordinates(self, x, y, z):
        """
        Set the feature value to end point at the given 3D coordinates.
        """
        self.set_feature_by_coordinates(x, y, z, self.END)

    def set_visited_by_coordinates(self, x, y, z):
        """
        Set the feature value to visited at the given 3D coordinates.
        """
        self.set_feature_by_coordinates(x, y, z, self.VISITED)

    def get_coordinates_by_feature(self, state, feature):
        """
        Retrieve all coordinates with a specified feature value.
        """
        matching_coordinates = [coord for coord, value in zip(self.coordinates, state) if value == feature]
        return matching_coordinates

    def is_point_inside_cone(self, x, y, z, apex, height, angle):
        """检查一个点是否在给定角度的圆锥内部"""
        apex = np.array(apex)
        point = np.array([x, y, z])
        vector = point - apex
        distance = np.linalg.norm(vector)

        if distance == 0:
            return False  # 顶点本身不算在内

        # 圆锥的方向向量，这里假设方向向下，即(0, 0, -1)
        cone_vector = np.array([0, 0, -1])

        # 计算向量之间的夹角
        dot_product = np.dot(vector, cone_vector)
        angle_cos = dot_product / (distance * np.linalg.norm(cone_vector))

        # 将角度从度转换为弧度
        angle_rad = np.radians(angle)

        return angle_cos >= np.cos(angle_rad)

    def calculate_3d_tangent_vector(self, A, B, center):
        """计算三维空间中圆上点B处的切线矢量"""
        A = np.array(A, dtype=float)
        B = np.array(B, dtype=float)
        center = np.array(center, dtype=float)

        # 从中心到 B 的向量
        vector_CB = B - center

        # 由A、B和中心定义的平面的法向量
        plane_normal = np.cross(A - center, B - center)
        if np.linalg.norm(plane_normal) == 0:
            return np.array([0, 0, 0])  # 避免除以零
        plane_normal = plane_normal / np.linalg.norm(plane_normal)

        # B处的切线矢量垂直于矢量CB和平面法线
        tangent_vector = np.cross(plane_normal, vector_CB)
        if np.linalg.norm(tangent_vector) == 0:
            return np.array([0, 0, 0])  # 避免除以零
        tangent_vector = tangent_vector / np.linalg.norm(tangent_vector)

        return tangent_vector

    def calculate_circle_center(self, A, B, radius=15):
        """计算以给定半径穿过A和B的圆心"""
        A = np.array(A, dtype=float)
        B = np.array(B, dtype=float)

        # 计算A和B之间的中点
        midpoint = (A + B) / 2

        # 从A到B的向量
        AB = B - A
        AB_length = np.linalg.norm(AB)

        if AB_length == 0:
            return A  # A和B是同一点

        # 计算中点到圆心的距离
        try:
            distance_to_center = math.sqrt(radius ** 2 - (AB_length / 2) ** 2)
        except ValueError:
            distance_to_center = 0  # 如果半径太小，无法形成圆

        # 中点到圆心的方向垂直于AB
        direction_to_center = np.cross(AB, [0, 0, 1])
        if np.linalg.norm(direction_to_center) == 0:  # 如果AB在Z轴方向
            direction_to_center = np.cross(AB, [0, 1, 0])
        if np.linalg.norm(direction_to_center) == 0:
            direction_to_center = np.array([0, 0, 1])  # 默认方向
        direction_to_center = direction_to_center / np.linalg.norm(direction_to_center)

        # 计算两个可能的圆心
        center1 = midpoint + distance_to_center * direction_to_center
        center2 = midpoint - distance_to_center * direction_to_center

        # 选择z轴较低的圆心
        if center1[2] < center2[2]:
            return center1
        else:
            return center2

    def index_to_coordinates(self, index):
        """
        Convert a one-dimensional index to a three-dimensional coordinate.
        """
        if 0 <= index < 1000:
            z = index // 100  # 计算z轴（每100个点一个z层）
            index %= 100
            y = index // 10  # 计算y轴（每10个点一个y层）
            x = index % 10   # 计算x轴（剩余的点为x轴）
            return x, y, z
        else:
            raise IndexError("Index out of bounds.")

    def calculate_reward_based_on_distance(self, point1, point2):
        """
        Calculate the reward based on the distance between two points in 3D space.
        The closer the points are, the higher the reward, with a maximum of 20.
        """
        if tuple(point2) == tuple(point1):
            return 20
        M = 20

        distance = np.linalg.norm(np.array(point1) - np.array(point2) + 1e-8)
        reward = np.clip(1 / distance, 0, M)
        return reward * 10  # Clamp the reward to the range [0-20]

    def distance_3d(self, points):
        """计算路径的总距离（基于奖励的距离计算）"""
        total_distance = 0
        for i in range(len(points) - 1):
            total_distance += self.calculate_reward_based_on_distance(points[i], points[i + 1])
        return total_distance * 5
    def get_integer_points_on_line(self, start_point, direction_vector, space_size=10):
        """
        Calculate integer points on a line given a starting point and a direction vector within a defined space.

        :param start_point: (tuple) Starting point coordinates (x, y, z).
        :param direction_vector: (np.array) Normalized direction vector.
        :param space_size: (int) Size of the cubic space.
        :return: (list) List of integer points on the line.
        """
        integer_points = []
        # Normalize the direction vector to have its smallest non-zero component be 1
        min_nonzero_component = np.min(np.abs(direction_vector[np.nonzero(direction_vector)]))
        step_vector = direction_vector / min_nonzero_component

        current_point = np.array(start_point, dtype=float)

        # Add the starting point if it's an integer within the bounds
        if np.all(np.mod(current_point, 1) == 0) and np.all(current_point >= 0) and np.all(current_point < space_size):
            integer_points.append(tuple(current_point.astype(int)))

        while True:
            # Move to the next point along the line
            current_point += step_vector

            # Round to the nearest integer to check if the new point is in bounds
            next_point = np.round(current_point).astype(int)

            # Check if the point is within the space bounds
            if np.all(next_point >= 0) and np.all(next_point < space_size):
                # Check if this integer point is a new discovery
                if not any(np.array_equal(next_point, p) for p in integer_points):
                    integer_points.append(tuple(next_point))
            else:
                # If we're out of bounds, stop the loop
                break
        if len(integer_points) > 1:
            integer_points = integer_points[1:]
        else:
            integer_points = []
        return integer_points
    def curve_point(self, cone_apex):
        """计算在45度圆锥曲面内部，同时在10度圆锥曲面外部的整数点"""
        contained_points_corrected = []
        for x in range(11):  # x from 0 to 10
            for y in range(11):  # y from 0 to 10
                for z in range(11):  # z from 0 to 10
                    # 检查点是否在45度圆锥内部
                    inside_45_cone = self.is_point_inside_cone(x, y, z, cone_apex, 10, 45)
                    # 检查点是否在10度圆锥外部
                    outside_10_cone = not self.is_point_inside_cone(x, y, z, cone_apex, 10, 10)
                    # 如果两个条件都满足，则记录点
                    if inside_45_cone and outside_10_cone:
                        contained_points_corrected.append((x, y, z))
        return contained_points_corrected

    def generate_circle_points(self, A, B, center, radius, num_points=100):
        A = np.array(A, dtype=float)
        B = np.array(B, dtype=float)
        center = np.array(center, dtype=float)
        vector_CA = A - center
        vector_CB = B - center
        normal_vector = np.cross(vector_CA, vector_CB)
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        angle = np.arccos(np.dot(vector_CA, vector_CB) / (np.linalg.norm(vector_CA) * np.linalg.norm(vector_CB)))
        angles = np.linspace(0, angle, num_points)
        circle_points = []
        for a in angles:
            rotation_matrix = (
                    np.cos(a) * np.eye(3) +
                    np.sin(a) * np.array([
                [0, -normal_vector[2], normal_vector[1]],
                [normal_vector[2], 0, -normal_vector[0]],
                [-normal_vector[1], normal_vector[0], 0]
            ]) +
                    (1 - np.cos(a)) * np.outer(normal_vector, normal_vector)
            )
            point = center + radius * np.dot(rotation_matrix, vector_CA / np.linalg.norm(vector_CA))
            circle_points.append(point)
        return np.array(circle_points)