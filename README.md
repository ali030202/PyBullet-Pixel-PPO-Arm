# PyBullet Pixel PPO Arm

一个用于桌面机械臂定点到达任务的轻量具身智能 demo。环境把 PyBullet 仿真画面作为观测，PPO 策略通过 CNN 从 RGB 像素输出两个关节的连续控制量，并按末端到目标的距离变化计算奖励。

## 闭环流程

仿真相机 RGB 像素 → CNN 特征提取 → PPO 连续策略 → 两个关节增量 → PyBullet 执行 → 到达奖励

观测只包含 84×84 RGB 图像。环境内部用真实末端位置计算奖励和成功标志，不把关节状态或目标坐标交给策略。

## 目录

- arm_rl/env.py：PyBullet 平面二关节机械臂、像素观测和到达奖励
- arm_rl/models.py：CNN actor-critic
- arm_rl/ppo.py：PyTorch PPO 与检查点保存
- assets/planar_arm.urdf：简易二关节机械臂模型
- train.py：固定目标训练
- adapt.py：改变目标位置与光照后继续微调
- evaluate.py：运行策略并统计成功率

## 安装

推荐 Python 3.10–3.12。

~~~bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
~~~

## 固定场景训练与评估

~~~bash
python train.py --steps 200000
python evaluate.py --checkpoint runs/ppo_pixel_arm.pt --episodes 20
~~~

训练会把策略保存到 runs/ppo_pixel_arm.pt。评估脚本输出到达成功率和平均距离。训练时长与硬件、随机种子和依赖版本有关；仓库不包含预训练权重。

## 视觉环境适配

用已训练策略在随机目标位置和亮度变化下继续训练：

~~~bash
python adapt.py --checkpoint runs/ppo_pixel_arm.pt --steps 50000
python evaluate.py --checkpoint runs/adapted_pixel_arm.pt --episodes 20 --randomize
~~~

## 任务定义

- 动作：两个关节各自的归一化连续角度增量
- 奖励：靠近目标获得距离进展奖励，成功到达获得额外奖励，每步有轻微代价
- 成功条件：末端与目标的平面距离小于 4.5 cm
- 适配：改变目标采样位置与相机渲染光照，再从原策略继续更新

这是教学用途的简化仿真 demo，不代表已验证的收敛曲线或真实机械臂性能。可参考 [RL-Robotics-Arm-PyBullet](https://github.com/ParanoidCode/RL-Robotics-Arm-PyBullet) 的 PyBullet 机械臂 PPO 任务；本仓库的环境和像素策略实现独立编写。
