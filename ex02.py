import cv2 as cv # OpenCV 라이브러리를 불러옵니다.
import numpy as np # 배열 연산을 위한 numpy 라이브러리를 불러옵니다.

img = np.full((500, 500, 3), 255, dtype=np.uint8) # 500x500 픽셀 크기의 흰색(255) 배경을 가진 3채널(BGR) 이미지를 생성합니다.

brush_size = 5 # 붓의 초기 크기를 5로 설정합니다.
is_drawing = False # 현재 마우스로 그림을 그리고 있는 상태인지 나타내는 상태 변수를 False로 초기화합니다.
color = (0, 0, 0) # 붓의 색상을 저장할 변수를 검은색으로 초기화합니다.

def draw_circle(event, x, y, flags, param): # 마우스 이벤트를 처리할 콜백 함수를 정의합니다.
    global is_drawing, color, brush_size # 함수 외부의 전역 변수들을 함수 내부에서 수정할 수 있도록 선언합니다.

    if event == cv.EVENT_LBUTTONDOWN: # 마우스 왼쪽 버튼이 눌렸을 때
        is_drawing = True # 그리기 상태를 True로 변경하여 그리기를 시작합니다.
        color = (255, 0, 0) # 붓의 색상을 파란색(BGR 기준 255, 0, 0)으로 설정합니다.
    elif event == cv.EVENT_RBUTTONDOWN: # 마우스 오른쪽 버튼이 눌렸을 때
        is_drawing = True # 그리기 상태를 True로 변경하여 그리기를 시작합니다.
        color = (0, 0, 255) # 붓의 색상을 빨간색(BGR 기준 0, 0, 255)으로 설정합니다.
    elif event == cv.EVENT_MOUSEMOVE: # 마우스가 움직일 때
        if is_drawing: # 만약 그리기 상태가 True라면 (버튼을 누른 채로 움직인다면)
            cv.circle(img, (x, y), brush_size, color, -1) # 현재 마우스 좌표(x,y)에 설정된 크기와 색상으로 내부가 채워진(-1) 원을 그립니다.
    elif event == cv.EVENT_LBUTTONUP or event == cv.EVENT_RBUTTONUP: # 마우스 왼쪽이나 오른쪽 버튼을 뗐을 때
        is_drawing = False # 그리기 상태를 False로 변경하여 그리기를 멈춥니다.
        cv.circle(img, (x, y), brush_size, color, -1) # 버튼을 뗀 마지막 위치에 원을 하나 그리고 마무리합니다.

cv.namedWindow('Painting') # 'Painting'이라는 이름의 윈도우 창을 생성합니다.
cv.setMouseCallback('Painting', draw_circle) # 'Painting' 창에 마우스 이벤트가 발생하면 draw_circle 함수를 실행하도록 콜백을 연결합니다.

while True: # 사용자가 종료할 때까지 화면을 계속 갱신하는 무한 루프를 엽니다.
    cv.imshow('Painting', img) # 'Painting' 창에 현재까지 그려진 캔버스(img) 이미지를 보여줍니다.

    key = cv.waitKey(1) & 0xFF # 1밀리초 동안 대기하며 키보드 입력을 받고, 그 값의 아스키코드를 추출합니다.

    if key == ord('q'): # 만약 입력된 키가 소문자 'q'라면
        break # 무한 루프를 빠져나가 프로그램을 종료합니다.
    elif key == ord('+') or key == ord('='): # 만약 입력된 키가 '+' 이거나 '=' 라면
        if brush_size < 15: # 현재 붓 크기가 최대 제한인 15보다 작을 때만
            brush_size += 1 # 붓 크기를 1 증가시킵니다.
    elif key == ord('-'): # 만약 입력된 키가 '-' 라면
        if brush_size > 1: # 현재 붓 크기가 최소 제한인 1보다 클 때만
            brush_size -= 1 # 붓 크기를 1 감소시킵니다.

cv.destroyAllWindows() # 열려있는 모든 창을 닫고 프로그램을 안전하게 종료합니다.