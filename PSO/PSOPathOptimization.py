import numpy as np
import math
import copy
import matplotlib
matplotlib.use('Agg')  # 或者其他合适的后端，如 'Qt5Agg', 'Agg' 等
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time  # 导入时间模块
class PSOPathOptimization:
    def __init__(self, env, coord, num_particles=50, max_iter=100, inertia=0.9, inertia_min=0.4, cognitive=1.5,
                 social=1.5, cone_height=5, cone_angle=45):
        """
        初始化PSO路径优化器
        :param env: GridEnvironment实例
        :param coord: Coordinate实例
        :param num_particles: 粒子数量
        :param max_iter: 最大迭代次数
        :param inertia: 惯性权重
        :param cognitive: 认知系数
        :param social: 社会系数
        :param cone_height: 圆锥高度
        :param cone_angle: 圆锥角度
        """
        self.env = env
        self.coord = coord
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.inertia = inertia
        self.inertia_min = inertia_min
        self.inertia_decay = (inertia - inertia_min) / max_iter
        self.cognitive = cognitive
        self.social = social
        self.cone_height = cone_height
        self.cone_angle = cone_angle

        # 每个粒子有四个控制点
        # P1的x和y固定为起点的x和y，仅优化P1的z
        # P2-P4优化x, y, z
        self.num_control_points = 4
        self.num_dimensions = 10  # [P1_z, P2_x, P2_y, P2_z, P3_x, P3_y, P3_z, P4_x, P4_y, P4_z]

        # 初始化粒子位置和速度
        # P1_z在0到起点z之间，P2-P4的x,y,z在0到9之间
        self.particles = np.zeros((self.num_particles, self.num_dimensions), dtype=int)
        for i in range(self.num_particles):
            # P1_z: 0到起点z
            self.particles[i, 0] = np.random.randint(0, self.env.start_coordinate[2] + 1)
            # P2-P4: x, y, z in 0-9
            self.particles[i, 1:] = np.random.randint(0, 10, self.num_dimensions - 1)

        # 初始化速度（浮点数，但后续位置会被四舍五入）
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, self.num_dimensions))

        # 记录个人最佳位置和适应度
        self.personal_best_positions = copy.deepcopy(self.particles)
        self.personal_best_scores = np.full(self.num_particles, np.inf)

        # 记录全局最佳位置和适应度
        self.global_best_position = None
        self.global_best_score = np.inf

        # 记录适应度历史
        self.fitness_history = []

    def calculate_euclidean_distance(self, point1, point2):
        """计算两点之间的欧氏距离"""
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def fitness_function(self, particle):
        """
        适应度函数：计算路径的总距离，并考虑避障和曲线区域约束
        :param particle: 粒子的位置信息（10维）
        :return: (适应度值, 惩罚原因列表)
        """
        # 提取控制点
        P1_z = particle[0]
        P1 = (self.env.start_coordinate[0], self.env.start_coordinate[1], P1_z)
        P2 = tuple(particle[1:4])
        P3 = tuple(particle[4:7])
        P4 = tuple(particle[7:10])

        # 构建完整路径：起点 -> P1 -> P2 -> P3 -> P4 -> 终点
        path = [
            self.env.start_coordinate,
            P1,  # P1
            P2,
            P3,
            P4,
            self.env.end_coordinate
        ]

        total_distance = 0
        penalty = 0
        penalty_details = []

        # 获取起点的 z 坐标
        start_z = self.env.start_coordinate[2]

        # **新增惩罚：P1_z 必须低于起点的 z 坐标**
        if P1_z >= start_z:
            penalty += 100  # P1_z 不满足条件的惩罚
            penalty_details.append(f"P1_z = {P1_z} 不低于起点 z 坐标 = {start_z}")

        # 定义哪些段是曲线段，这里假设第二段和第四段是曲线
        # 路径分为五段，索引从0到4，对应时间段1到5
        curve_segments = [1, 3]  # 时间段2和4

        for i in range(len(path) - 1):
            p_start = path[i]
            p_end = path[i + 1]

            # 计算当前路径段的距离
            segment_distance = self.calculate_euclidean_distance(p_start, p_end)
            total_distance += segment_distance

            if i == len(path) - 2:
                # 最后一段（P4 到 P5），不进行任何惩罚检查
                continue

            # 检查控制点是否与障碍物重叠
            if p_end in self.env.special_coordinates:
                penalty += 100  # 碰撞惩罚
                penalty_details.append(f"粒子控制点 P{i + 2} 与障碍物重叠 at {p_end}")
                continue  # 跳过曲线区域检查

            # 检查曲线段是否满足几何约束
            if i in curve_segments:
                # 获取曲线段的有效点列表
                contained_points_corrected = self.coord.curve_point(p_start)

                # 检查 P_end 是否在有效点列表中
                if p_end not in contained_points_corrected:
                    penalty += 50  # 不在两个曲面之间
                    penalty_details.append(f"粒子控制点 P{i + 2} 不在两个曲面之间 at {p_end}")

                # 检查 P_end 是否在圆锥内
                if not self.coord.is_point_inside_cone(
                        x=p_end[0],
                        y=p_end[1],
                        z=p_end[2],
                        apex=p_start,
                        height=self.cone_height,
                        angle=self.cone_angle
                ):
                    penalty += 50  # 圆锥外惩罚
                    penalty_details.append(f"粒子控制点 P{i + 2} 不在圆锥内 at {p_end}")
                    # **新增惩罚：确保 P3 位于由 P1 和 P2 生成的整数点列表中**
                    # 计算从 P1 到 P2 的切向量和整数点列表
            if i == 1:
                try:
                    circle_center = self.coord.calculate_circle_center(P1, P2)
                    tangent_vector = self.coord.calculate_3d_tangent_vector(P1, P2, circle_center)
                    integer_points = self.coord.get_integer_points_on_line(P2, tangent_vector)
                    if P3 not in integer_points:
                        penalty += 75  # P3 不在整数点列表中的惩罚
                        penalty_details.append(f"P3 = {P3} 不在由 P1 和 P2 生成的整数点列表中")
                except Exception as e:
                    # 如果计算过程中出现错误，给予较大的惩罚
                    penalty += 100
                    penalty_details.append(f"计算 P3 的整数点列表时出错: {e}")

        # 适应度 = 总距离 + 惩罚
        fitness = total_distance + penalty
        return fitness, penalty_details

    def update_velocity_position(self):
        """更新粒子的速度和位置，确保位置为整数"""
        for i in range(self.num_particles):
            # 更新速度
            r1 = np.random.rand(self.num_dimensions)
            r2 = np.random.rand(self.num_dimensions)

            cognitive_velocity = self.cognitive * r1 * (self.personal_best_positions[i] - self.particles[i])
            social_velocity = self.social * r2 * (self.global_best_position - self.particles[i])
            self.velocities[i] = self.inertia * self.velocities[i] + cognitive_velocity + social_velocity

            # 更新位置
            new_position = self.particles[i] + self.velocities[i]

            # 四舍五入为整数
            new_position = np.round(new_position).astype(int)

            # 确保 P1 的 x 和 y 坐标固定为起点的 x 和 y
            # 这里只需保持 P1 的 x 和 y 与起点一致，P1_z 已经在粒子中表示
            # 由于 P1 的 x 和 y 固定为起点的 x 和 y，不需要在粒子中存储，因此无需调整

            # 确保所有坐标在0到9之间
            new_position = np.clip(new_position, 0, 9)

            # 更新粒子的位置
            self.particles[i] = new_position

    def optimize(self):
        total_time = 0  # 初始化累计时间
        """运行PSO算法进行路径优化"""
        for iter_num in range(self.max_iter):
            start_time = time.time()  # 记录本轮开始时间
            fitness_values = []
            penalty_details_all = []  # 存储所有粒子的惩罚详情

            for i in range(self.num_particles):
                particle = self.particles[i]
                fitness, penalty_details = self.fitness_function(particle)
                fitness_values.append(fitness)
                penalty_details_all.append(penalty_details)

                # 更新个人最佳
                if fitness < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = fitness
                    self.personal_best_positions[i] = particle.copy()

                # 更新全局最佳
                if fitness < self.global_best_score:
                    self.global_best_score = fitness
                    self.global_best_position = particle.copy()

            # 记录适应度历史
            self.fitness_history.append(self.global_best_score)

            # 获取当前全局最佳路径的控制点
            if self.global_best_position is not None:
                P1_z = self.global_best_position[0]
                P1 = (self.env.start_coordinate[0], self.env.start_coordinate[1], P1_z)
                P2 = tuple(self.global_best_position[1:4])
                P3 = tuple(self.global_best_position[4:7])
                P4 = tuple(self.global_best_position[7:10])
                best_path = [self.env.start_coordinate, P1, P2, P3, P4, self.env.end_coordinate]
            else:
                best_path = []

            # 打印当前迭代的适应度和路径控制点
            print(f"迭代 {iter_num + 1}/{self.max_iter}")
            print(f"全局最佳适应度: {self.global_best_score}")
            print(f"全局最佳路径控制点位置: {best_path}\n")

            # 打印所有粒子的适应度和路径控制点
            print("所有粒子的适应度和路径控制点：")
            for idx, (particle, fitness, penalties) in enumerate(
                    zip(self.particles, fitness_values, penalty_details_all)):
                P1_z = particle[0]
                P1 = (self.env.start_coordinate[0], self.env.start_coordinate[1], P1_z)
                P2 = tuple(particle[1:4])
                P3 = tuple(particle[4:7])
                P4 = tuple(particle[7:10])
                path = [self.env.start_coordinate, P1, P2, P3, P4, self.env.end_coordinate]
                print(f"粒子 {idx + 1}: 适应度 = {fitness}, 控制点 = {path}")
                if penalties:
                    print("  惩罚详情:")
                    for penalty in penalties:
                        print(f"    - {penalty}")
                else:
                    print("  无惩罚")
                print("=" * 50)  # 分隔线，便于阅读

            print("=" * 100)  # 分隔线，便于阅读


            end_time = time.time()  # 记录本轮结束时间
            iteration_time = end_time - start_time  # 计算本轮时间
            total_time += iteration_time  # 累加总时间

            # 打印本轮和累计时间
            print(f"本轮迭代时间: {iteration_time:.2f} 秒")
            print(f"累计时间: {total_time:.2f} 秒")
            print("=" * 100)  # 再次分隔线，便于阅读

            # 更新粒子的速度和位置
            self.update_velocity_position()

            # 动态调整惯性权重
            self.inertia = max(self.inertia_min, self.inertia - self.inertia_decay)

        return self.global_best_position, self.global_best_score

    def visualize_path(self, best_position):
        """可视化优化后的路径"""
        if best_position is None:
            print("未找到最佳路径。")
            return

        # 提取控制点
        P1_z = best_position[0]
        P1 = (self.env.start_coordinate[0], self.env.start_coordinate[1], P1_z)
        P2 = tuple(best_position[1:4])
        P3 = tuple(best_position[4:7])
        P4 = tuple(best_position[7:10])

        # 构建完整路径
        path = [
            P1,
            P2,
            P3,
            P4,
            self.env.end_coordinate
        ]

        # 计算路径段之间的欧氏距离
        full_discrete_path = []
        for i in range(len(path) - 1):
            p_start = path[i]
            p_end = path[i + 1]
            # 这里只绘制控制点之间的直线
            full_discrete_path.append(p_start)
            full_discrete_path.append(p_end)

        # 绘制路径和障碍物
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 绘制障碍物
        if self.env.special_coordinates:
            obs_x, obs_y, obs_z = zip(*self.env.special_coordinates)
            ax.scatter(obs_x, obs_y, obs_z, c='red', marker='s', label='障碍物')

        # 绘制路径
        if full_discrete_path:
            path_x, path_y, path_z = zip(*full_discrete_path)
            ax.plot(path_x, path_y, path_z, c='blue', label='优化路径')

        # 绘制起点和终点
        ax.scatter(*self.env.start_coordinate, c='green', s=100, label='起点')
        ax.scatter(*self.env.end_coordinate, c='purple', s=100, label='终点')

        ax.set_xlabel('X 轴')
        ax.set_ylabel('Y 轴')
        ax.set_zlabel('Z 轴')
        ax.set_title('使用PSO优化的路径')
        ax.legend()
        plt.show()

    def plot_fitness_history(self):
        """绘制适应度历史图"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.fitness_history, label='最佳适应度')
        plt.xlabel('迭代次数')
        plt.ylabel('适应度')
        plt.title('PSO适应度演化')
        plt.legend()
        plt.grid()
        plt.show()
