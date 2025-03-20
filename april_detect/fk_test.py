import numpy as np

# DH 파라미터 (URDF 기반)
dh_params = [
    {'theta': 0,   'd': 0.089, 'a': 0,    'alpha': 0},  # Joint0
    {'theta': 0,   'd': 0.06,  'a': 0.02, 'alpha': -np.pi/2},        # Joint1
    {'theta': 0,   'd': 0, 'a': 0.0,    'alpha': np.pi/2},
    {'theta': 0,   'd': 0.096, 'a': 0.0,    'alpha': 0}, # Joint2
    {'theta': 0,   'd': 0.06,  'a': 0,    'alpha': -np.pi/2},  # Joint3
    {'theta': 0,   'd': 0.00, 'a': 0.0,    'alpha': np.pi/2},
    {'theta': 0,   'd': 0.094, 'a': 0,    'alpha': -np.pi/2}, # Joint4
    {'theta': 0,   'd': 0.00, 'a': 0.0,    'alpha': np.pi/2},
    {'theta': 0,   'd': 0.095, 'a': 0.02, 'alpha': 0},        # Joint5
]

# 관절 각도 제한 (라디안)
joint_limits = [
    (0, 3*np.pi/2),          # Joint0
    (-3*np.pi/4, 3*np.pi/4), # Joint1
    (-3*np.pi/4, 3*np.pi/4), # Joint2
    (-3*np.pi/4, 3*np.pi/4), # Joint3
    (-3*np.pi/4, 3*np.pi/4), # Joint4
    (-np.pi/2, np.pi/2)      # Joint5
]

def dh_matrix(theta, d, a, alpha):
    """단일 관절에 대한 DH 변환 행렬 생성"""
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,      sa,     ca,    d],
        [0,       0,      0,    1]
    ])

def forward_kinematics(joint_angles):
    """
    관절 각도를 입력받아 엔드 이펙터 위치 계산
    :param joint_angles: 6개 관절 각도 (라디안)
    :return: [x, y, z] 위치 (m)
    """
    # 초기 변환 행렬 (기본 좌표계)
    T = np.eye(4)
    
    # 각 관절에 대해 DH 행렬 누적 계산
    for i in range(len(dh_params[0])):
        # DH 파라미터 업데이트
        params = dh_params[i].copy()
        params['theta'] += joint_angles[i]  # 관절 각도 적용
        
        # 개별 변환 행렬 계산
        Ti = dh_matrix(**params)
        
        # 전체 변환 행렬 업데이트
        T = T @ Ti
    
    # 엔드 이펙터 위치 추출
    return T[:3, 3]

# 테스트 케이스 (모든 관절 0도)
angles = [0, 0, 0, 0, 0, 0]
print("Home position:", forward_kinematics(angles))

# Joint1을 90도 회전
angles = [0, np.pi/2, 0, 0, 0, 0]
print("Joint1 90°:", forward_kinematics(angles))
# ✅ 테스트 실행 (예제 관절 각도)
#joint_angles = [0, -35,0,-54,-65,0]  # [θ1, θ2, θ3, θ4, θ5, θ6]

#end_effector_position = forward_kinematics(joint_angles)
#print(f"✅ FK 결과 (엔드 이펙터 위치): {end_effector_position}")
