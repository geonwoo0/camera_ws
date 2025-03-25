import cv2
import time
import queue

def main():
    cam_index = 0  # 카메라 인덱스 0번 사용
    cap = cv2.VideoCapture(cam_index,cv2.CAP_V4L2)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2048)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1536)
    cap.set(cv2.CAP_PROP_FPS, 30)  # 15 FPS 설정

    if not cap.isOpened():
        print("카메라 열기 실패")
        return

    frame_queue = queue.Queue()
    prev_capture_time = None

    while True:
        # 프레임 읽기 시작 시간
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("프레임 읽기 실패")
            continue
        # 프레임 읽기 종료 시간
        end = time.time()
        capture_duration = (end - start) * 1000  # ms 단위
        print(f"프레임 캡처 소요 시간: {capture_duration:.2f} ms")

        # 프레임을 큐에 넣기 시작 시간
        put_start = time.time()
        frame_queue.put((time.time(), frame))
        put_end = time.time()
        put_duration = (put_end - put_start) * 1000  # ms 단위
        print(f"프레임 큐에 넣는 소요 시간: {put_duration:.2f} ms")

        if prev_capture_time is not None:
            interval = (start - prev_capture_time) * 1000
            print(f"프레임 간 간격: {interval:.2f} ms")
        prev_capture_time = start

        #cv2.imshow("Frame", frame)
        #if cv2.waitKey(1) & 0xFF == 27:  # ESC 키 종료
        #    break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
