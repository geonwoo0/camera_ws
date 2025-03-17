import cv2

def check_index(max_index=15):
    available_camera = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"카메라 인덱스 {i} 사용가능")
            available_camera.append(i)
            cap.release()
        else:
            print(f"카메라 인덱스 {i} 사용불가")
    return available_camera

if __name__ == "__main__":
    available = check_index()
    print("사용 가능 카메라 인덱스 : ",available)