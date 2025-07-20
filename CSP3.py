import cv2
import mediapipe as mp
import pyautogui
import speech_recognition as sr

# Initialize mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Function to detect "pinch" gesture
def detect_pinch(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    # Calculate distance between thumb and index finger
    pinch_distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
    return pinch_distance < 0.03  # Threshold for pinch gesture

# Function to switch between tabs
def switch_tabs():
    pyautogui.hotkey("alt", "tab")

# Function to start gesture-based tab switching
def start_tab_switching():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip and process the frame
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if detect_pinch(hand_landmarks):
                    print("Pinch detected: Switching tabs.")
                    switch_tabs()

        # Display the video feed
        cv2.imshow("Pinch Gesture Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to listen for the "Switch Tab" voice command
def listen_for_command():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("Listening for the command 'Switch Tab'...")
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            command = recognizer.recognize_google(audio).lower()
            print(f"Command received: {command}")
            if "switch tab" in command:
                print("Starting pinch gesture-based tab switching...")
                start_tab_switching()
            else:
                print("Invalid command. Please say 'Switch Tab' to start.")
    except sr.UnknownValueError:
        print("Could not understand the audio.")
    except sr.RequestError:
        print("Could not request results; check your network connection.")

# Run the voice recognition system
if __name__ == "__main__":
    listen_for_command()
