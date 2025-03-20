import cv2
import apriltag
import time
import numpy as np

def main():
    # 두 대의 카메라(인덱스 0, 1) 캡쳐 객체 생성
    cap0 = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap1 = cv2.VideoCapture(2, cv2.CAP_V4L2)

    # MJPG 코덱 설정 및 해상도, FPS 설정
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    for cap in [cap0, cap1]:
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)

    # 에이프릴 태그 디텍터 생성
    detector = apriltag.Detector()

    frame_timestamp0 = None  # 카메라 0 프레임 타임스탬프
    frame_timestamp1 = None  # 카메라 1 프레임 타임스탬프

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        if not ret0 or not ret1:
            print("하나 이상의 카메라에서 프레임을 읽어오지 못했습니다.")
            break

        # 각 카메라에서 프레임 캡쳐 시각 기록
        frame_timestamp0 = time.time()
        frame_timestamp1 = time.time()

        # --- 카메라 0 처리 ---
        gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        results0 = detector.detect(gray0)
        for r in results0:
            pts = np.int32(r.corners)
            cv2.polylines(frame0, [pts.reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 0), thickness=2)
            center = (int(r.center[0]), int(r.center[1]))
            cv2.putText(frame0, str(r.tag_id), center, cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)

        # --- 카메라 1 처리 ---
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        results1 = detector.detect(gray1)
        for r in results1:
            pts = np.int32(r.corners)
            cv2.polylines(frame1, [pts.reshape((-1, 1, 2))], isClosed=True, color=(0, 255, 0), thickness=2)
            center = (int(r.center[0]), int(r.center[1]))
            cv2.putText(frame1, str(r.tag_id), center, cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2)

        # 결과 프레임 별도 창에 출력
        cv2.imshow("AprilTag Detection - Camera 0", frame0)
        cv2.imshow("AprilTag Detection - Camera 1", frame1)

        # 키 입력 대기 (1ms)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # 엔터(Enter) 키: 각 카메라의 프레임 캡쳐와 현재 시간 차이 출력
            current_timestamp = time.time()
            time_diff0 = current_timestamp - frame_timestamp0
            time_diff1 = current_timestamp - frame_timestamp1
            print("카메라 0: 프레임 타임스탬프와 현재 타임스탬프 차이: {:.3f} 초".format(time_diff0))
            print("카메라 1: 프레임 타임스탬프와 현재 타임스탬프 차이: {:.3f} 초".format(time_diff1))
        elif key == 27:  # ESC 키: 종료
            break

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
