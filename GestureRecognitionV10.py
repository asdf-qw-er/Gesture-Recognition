import cv2
import mediapipe as mp

# MediaPipe 손 모델 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def detect_finger_states(hand_landmarks, hand_label, is_back_hand):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []

    # 엄지손가락
    if hand_label == "Right":
        if is_back_hand:
            if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0] - 1].x:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
                fingers.append(1)
            else:
                fingers.append(0)
    else:  # Left hand
        if is_back_hand:
            if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0] - 1].x:
                fingers.append(1)
            else:
                fingers.append(0)

    # 나머지 손가락
    for id in range(1, 5):
        if is_back_hand:
            if hand_landmarks.landmark[tip_ids[id]].y > hand_landmarks.landmark[tip_ids[id] - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)
        else:
            if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)
            
    return fingers

def detect_gesture(finger_states):
    # 주먹
    if finger_states == [0, 0, 0, 0, 0]:
        return 'Fist'
    
    # 편 손
    elif finger_states == [1, 1, 1, 1, 1]:
        return 'Open Hand'
    
    # 좋아요 (엄지 손가락만 펴져 있음)
    elif finger_states == [1, 0, 0, 0, 0]:
        return 'Thumbs Up'
    
    # 욕 (중지만 펴져 있음)
    elif finger_states == [0, 0, 1, 0, 0]:
        return 'FUCK YOU'
    
    # OK 제스처 (엄지와 검지가 만남)
    elif finger_states == [1, 1, 0, 0, 0]:
        return 'OK'
    
    # 평화 제스처 (검지와 중지만 펴져 있음)
    elif finger_states == [0, 1, 1, 0, 0]:
        return 'Peace'
    
    # 1
    elif finger_states == [0, 1, 0, 0, 0]:
        return 'One'

    # 3
    elif finger_states == [0, 1, 1, 1, 0]:
        return 'Three'
    
    # 4
    elif finger_states == [0, 1, 1, 1, 1]:
        return 'Four'
    
    # 니코니코니 제스처
    elif finger_states == [1, 1, 0, 0, 1]:
        return 'Niko Niko Ni'
    
    return 'Unknown'

def is_back_hand(hand_landmarks, hand_label):
    # 손목과 중지 중심을 이용해 손등/손바닥을 구분
    wrist = hand_landmarks.landmark[0]
    middle_finger_mcp = hand_landmarks.landmark[9]
    
    if hand_label == "Left":
        if wrist.x > middle_finger_mcp.x:
            return True
        else:
            return False
    else:  # Right hand
        if wrist.x < middle_finger_mcp.x:
            return True
        else:
            return False

# 비디오 캡처 초기화
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # BGR 이미지를 RGB로 변환
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 성능을 높이기 위해 이미지 작동 설정 (기본값 True)
    image.flags.writeable = False
    
    # 손가락 랜드마크 탐지
    results = hands.process(image)
    
    # 이미지를 다시 writeable로 설정하여 손가락 랜드마크 그리기
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_hand_landmarks:
        for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            # 손가락 랜드마크 그리기
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 손등 여부 판단
            hand_label = handedness.classification[0].label
            back_hand = is_back_hand(hand_landmarks, hand_label)
            
            # 손가락 상태 감지
            finger_states = detect_finger_states(hand_landmarks, hand_label, back_hand)
            
            # 제스처 인식
            gesture = detect_gesture(finger_states)
            
            # 왼손 또는 오른손 텍스트
            label_text = f'{hand_label} {"Back" if back_hand else "Palm"}: {gesture}'
            
            # 인식된 제스처 화면에 표시
            position = (10, 70 + (idx * 50)) # 여러 손에 대해 위치 조정
            cv2.putText(image, label_text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    # 프레임을 표시
    cv2.imshow('Hand Gesture Recognition', image)
    
    # q를 누르면 루프 종료
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()