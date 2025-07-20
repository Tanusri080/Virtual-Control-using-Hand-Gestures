import cv2
import mediapipe as mp
import pyautogui
import speech_recognition as sr
import threading


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


current_command = "none"
command_lock = threading.Lock()

def detect_fist(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    return all(
        ((tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2) ** 0.5 < 0.15
        for tip in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
    )


def control_volume(hand_landmarks, frame_width, frame_height):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    x1, y1 = int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height)
    x2, y2 = int(index_tip.x * frame_width), int(index_tip.y * frame_height)
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    if distance > 100:
        pyautogui.press("volumeup")
        print("Volume Up")
    elif distance < 50:
        pyautogui.press("volumedown")
        print("Volume Down")


def detect_pinch(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
  
    pinch_distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
    print(f"Pinch distance: {pinch_distance}")  
    
    return pinch_distance < 0.03  


def detect_swipe(hand_landmarks, prev_index_y):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    y_movement = prev_index_y - index_tip.y

    if y_movement > 0.03:
        return "up", index_tip.y
    elif y_movement < -0.03:
        return "down", index_tip.y
    return None, index_tip.y


def gesture_control():
    global current_command
    cap = cv2.VideoCapture(0)
    prev_index_y = 0  

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        with command_lock:
            command = current_command

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if command == "stop application" and detect_fist(hand_landmarks):
                    print("Fist detected: Stopping application...")
                    pyautogui.hotkey('alt', 'f4')

                elif command == "control volume":
                    control_volume(hand_landmarks, frame_width, frame_height)

                elif command == "switch tab" and detect_pinch(hand_landmarks):
                    print("Pinch detected: Switching tabs...")
                    pyautogui.hotkey("alt", "tab")

                elif command == "scroll":
                    swipe_direction, prev_index_y = detect_swipe(hand_landmarks, prev_index_y)
                    if swipe_direction == "up":
                        print("Swipe Up: Scrolling up...")
                        pyautogui.scroll(30)
                    elif swipe_direction == "down":
                        print("Swipe Down: Scrolling down...")
                        pyautogui.scroll(-30)

        cv2.putText(frame, f"Active Command: {command}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def listen_for_command():
    global current_command
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("Voice Command Listener active...")
    while True:
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
                print("Listening for command...")
                audio = recognizer.listen(source, timeout=5)  
                command = recognizer.recognize_google(audio).lower()
                print(f"Command received: {command}")
                
                valid_commands = ["stop application", "control volume", "switch tab", "scroll"]
                if command in valid_commands:
                    with command_lock:
                        current_command = command
                        print(f"Switched to: {current_command}")
                else:
                    print(f"Invalid command received: {command}. Try again.")
        except sr.UnknownValueError:
            print("Could not understand the audio. Please speak clearly.")
        except sr.RequestError as e:
            print(f"Error in request; check your network connection: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")


if __name__ == "__main__":

    gesture_thread = threading.Thread(target=gesture_control, daemon=True)
    voice_thread = threading.Thread(target=listen_for_command, daemon=True)

 
    gesture_thread.start()
    voice_thread.start()

   
    gesture_thread.join()
    voice_thread.join()
