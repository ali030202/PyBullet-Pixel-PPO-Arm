from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data


class PixelReachEnv:
    """Tiny PyBullet reaching task; policy observations contain pixels only."""

    def __init__(
        self,
        image_size=84,
        render=False,
        randomize_target=False,
        randomize_light=False,
        seed=None,
        max_steps=80,
    ):
        self.image_size = int(image_size)
        self.gui = bool(render)
        self.randomize_target = bool(randomize_target)
        self.randomize_light = bool(randomize_light)
        self.max_steps = int(max_steps)
        self.rng = np.random.default_rng(seed)
        self.client = p.connect(p.GUI if self.gui else p.DIRECT)
        self.urdf_path = Path(__file__).resolve().parent.parent / "assets" / "planar_arm.urdf"
        if not self.urdf_path.exists():
            raise FileNotFoundError(self.urdf_path)
        self.joint_ids = [0, 1]
        self.joint_low = np.array([-3.05, -2.8], dtype=np.float32)
        self.joint_high = np.array([3.05, 2.8], dtype=np.float32)
        self.max_joint_delta = 0.14
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        p.resetSimulation(physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.client)
        p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.arm = p.loadURDF(
            str(self.urdf_path),
            basePosition=[0, 0, 0.0],
            useFixedBase=True,
            physicsClientId=self.client,
        )
        for joint_id, angle in zip(self.joint_ids, [0.0, 0.0]):
            p.resetJointState(self.arm, joint_id, angle, physicsClientId=self.client)
        if self.randomize_target:
            self.target_xy = np.array(
                [self.rng.uniform(0.22, 0.45), self.rng.uniform(-0.23, 0.23)],
                dtype=np.float32,
            )
        else:
            self.target_xy = np.array([0.36, 0.10], dtype=np.float32)
        target_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.035,
            rgbaColor=[0.95, 0.22, 0.10, 1.0],
            physicsClientId=self.client,
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=target_visual,
            basePosition=[float(self.target_xy[0]), float(self.target_xy[1]), 0.035],
            physicsClientId=self.client,
        )
        self.light_color = (
            self.rng.uniform(0.55, 1.0, size=3).tolist()
            if self.randomize_light
            else [1.0, 1.0, 1.0]
        )
        self.light_direction = (
            self.rng.uniform(-1.0, 1.0, size=3).tolist()
            if self.randomize_light
            else [1.0, 1.0, 2.0]
        )
        self.steps = 0
        self.previous_distance = self._distance()
        return self._observation()

    def _end_effector_xy(self):
        state = p.getLinkState(
            self.arm,
            2,
            computeForwardKinematics=True,
            physicsClientId=self.client,
        )
        return np.asarray(state[4][:2], dtype=np.float32)

    def _distance(self):
        return float(np.linalg.norm(self._end_effector_xy() - self.target_xy))

    def _observation(self):
        eye = [0.27, 0.0, 1.05]
        target = [0.27, 0.0, 0.02]
        view = p.computeViewMatrix(eye, target, [0, 1, 0])
        projection = p.computeProjectionMatrixFOV(
            fov=48,
            aspect=1.0,
            nearVal=0.1,
            farVal=3.0,
        )
        image = p.getCameraImage(
            self.image_size,
            self.image_size,
            viewMatrix=view,
            projectionMatrix=projection,
            lightDirection=self.light_direction,
            lightColor=self.light_color,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.client,
        )[2]
        rgb = np.asarray(image, dtype=np.uint8).reshape(
            self.image_size, self.image_size, 4
        )[:, :, :3]
        return np.moveaxis(rgb, -1, 0).astype(np.float32) / 255.0

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(2)
        action = np.clip(action, -1.0, 1.0)
        current = np.asarray(
            [p.getJointState(self.arm, joint_id, physicsClientId=self.client)[0]
             for joint_id in self.joint_ids],
            dtype=np.float32,
        )
        target_positions = np.clip(
            current + self.max_joint_delta * action,
            self.joint_low,
            self.joint_high,
        )
        p.setJointMotorControlArray(
            self.arm,
            self.joint_ids,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_positions.tolist(),
            forces=[12.0, 10.0],
            positionGains=[0.35, 0.35],
            velocityGains=[1.0, 1.0],
            physicsClientId=self.client,
        )
        for _ in range(12):
            p.stepSimulation(physicsClientId=self.client)
        self.steps += 1
        distance = self._distance()
        success = distance < 0.045
        reward = 6.0 * (self.previous_distance - distance) - 0.01
        if success:
            reward += 1.0
        self.previous_distance = distance
        done = bool(success or self.steps >= self.max_steps)
        info = {"distance": distance, "success": bool(success)}
        return self._observation(), float(reward), done, info

    def close(self):
        if p.isConnected(self.client):
            p.disconnect(physicsClientId=self.client)
