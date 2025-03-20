import ikfast_binding
import math
import numpy as np
import random

def five_d_pose_to_six_d_pose(pose_5d):
    """
    5D pose: [x, y, z, roll, pitch]
    -> 6D pose: [x, y, z] + 9원소 회전행렬 (row-major)
    yaw는 제어 불가능하므로 0으로 가정합니다.
    """
    x, y, z, roll, pitch = pose_5d
    yaw = 0.0  # yaw 기본값

    # ZYX Euler 각도로 회전행렬 계산: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cr = math.cos(roll)
    sr = math.sin(roll)

    R = np.array([
        [cy * cp,             cy * sp * sr - sy * cr,   cy * sp * cr + sy * sr],
        [sy * cp,             sy * sp * sr + cy * cr,   sy * sp * cr - cy * sr],
        [-sp,                 cp * sr,                  cp * cr]
    ])

    # 6D pose = [x, y, z] + 회전행렬의 9원소 (행 우선 순서)
    pose_6d = [x, y, z] + R.flatten().tolist()
    return pose_6d


test_poses = []
for i in range(50):
    # 위치 범위: -0.3 ~ 0.3
    x = random.uniform(-0.3, 0.3)
    y = random.uniform(-0.3, 0.3)
    z = random.uniform(-0.3, 0.3)
    
    # 각도 범위: -30° ~ 30° -> 라디안 변환
    roll = random.uniform(-30, 30) * math.pi / 180.0
    pitch = random.uniform(-30, 30) * math.pi / 180.0
    
    test_poses.append([x, y, z, roll, pitch])

for p in test_poses:
    pose_6d = five_d_pose_to_six_d_pose(p)
    try:
        sol = ikfast_binding.compute_ik(pose_6d)
        print("Pose:", p, "-> IK 솔루션:", sol)
    except Exception as e:
        print("Pose:", p, "-> IK 계산 실패:", e)

# 예시: 5D pose 설정
pose_5d = [0.1, -0.1, 0.2, math.radians(10), math.radians(15)]
pose_6d = five_d_pose_to_six_d_pose(pose_5d)

try:
    solution = ikfast_binding.compute_ik(pose_6d)
    print("IK 솔루션:", solution)
except Exception as e:
    print("IK 계산 실패:", e)

# 예시: 관절 값 벡터 (관절 수는 GetNumJoints()에 의해 결정됨)
joints = [0.00000000e+00, 2.65513479e-01, 2.35549137e+00, 4.44004731e+00, 1.54039271e-03]
try:
    fk_pose = ikfast_binding.compute_fk(joints)
    print("FK 결과:", fk_pose)
except Exception as e:
    print("FK 계산 실패:", e)
