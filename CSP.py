import cv2
import mediapipe as mp
import pyautogui
import speech_recognition as sr

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Function to detect a "fist" gesture (closed hand)
def detect_fist(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    # Check if all fingertips are close to the wrist (closed fist)
    fist_detected = all(
        ((tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2) ** 0.5 < 0.15
        for tip in [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
    )
    return fist_detected

# Function to stop an active application (Alt+F4)
def stop_application():
    pyautogui.hotkey('alt', 'f4')  # Close active window
    print("Alt+F4 triggered: Application stopped.")

# Function to start gesture-based application stopping
def start_application_control():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirrored view
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Check for fist gesture
                if detect_fist(hand_landmarks):
                    print("Fist detected: Stopping application...")
                    stop_application()

        # Display the video feed
        cv2.imshow("Application Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to listen for voice commands
def listen_for_command():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("Listening for the command 'Detect App'...")
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            command = recognizer.recognize_google(audio).lower()
            print(f"Command received: {command}")
            if "detect app" in command:
                print("Starting gesture-based application control...")
                start_application_control()
            else:
                print("Invalid command. Please say 'Detect App' to start.")
    except sr.UnknownValueError:
        print("Could not understand the audio.")
    except sr.RequestError:
        print("Could not request results; check your network connection.")

# Run the voice recognition system
if __name__ == "__main__":
    listen_for_command()

