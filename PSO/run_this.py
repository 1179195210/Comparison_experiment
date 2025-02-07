from GridEnvironment import GridEnvironment
from Coordinate import Coordinate
from PSOPathOptimization import PSOPathOptimization

if __name__ == '__main__':
    # 创建Coordinate实例
    coor = Coordinate()

    # 创建GridEnvironment实例
    env = GridEnvironment()

    # 定义起点和终点坐标
    start = (6, 6, 9)  # 您原始代码中的起点
    end = (1, 3, 0)    # 您原始代码中的终点

    # 重置环境，初始化起点和终点
    env.reset(start, end)

    # 创建PSOPathOptimization实例
    pso_optimizer = PSOPathOptimization(
        env=env,
        coord=coor,
        num_particles=300,      # 粒子数量，可以根据需要调整
        max_iter=200,          # 迭代次数，可以根据需要调整
        inertia=1.2,
        cognitive=2.0,  # 增加认知系数
        social=1.0,  # 减少社会系数
        cone_height=5,         # 圆锥高度，可以根据需要调整
        cone_angle=45          # 圆锥角度，可以根据需要调整
    )

    # 运行优化
    best_position, best_score = pso_optimizer.optimize()

    # 输出最优解
    print("\n最优路径控制点位置：")
    P1_z = best_position[0]
    P1 = (env.start_coordinate[0], env.start_coordinate[1], P1_z)
    P2 = tuple(best_position[1:4])
    P3 = tuple(best_position[4:7])
    P4 = tuple(best_position[7:10])
    print(f"P1: {P1}")
    print(f"P2: {P2}")
    print(f"P3: {P3}")
    print(f"P4: {P4}")
    print(f"最优路径总距离: {best_score:.2f}")

    # 可视化适应度历史
    pso_optimizer.plot_fitness_history()

    # 可视化优化后的路径
    pso_optimizer.visualize_path(best_position)
