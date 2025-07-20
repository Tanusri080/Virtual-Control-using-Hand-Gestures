import cv2
import mediapipe as mp
import pyautogui
import speech_recognition as sr

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Variables to track gesture points
x1 = y1 = x2 = y2 = 0

# Function to control volume based on thumb and index finger distance
def control_volume(hand_landmarks):
    global x1, y1, x2, y2
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    # Calculate distance between thumb and index finger
    x1, y1 = int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height)
    x2, y2 = int(index_tip.x * frame_width), int(index_tip.y * frame_height)
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    # Control volume based on distance
    if distance > 100:  # Threshold for increasing volume
        pyautogui.press("volumeup")
        print("Volume Up")
    elif distance < 50:  # Threshold for decreasing volume
        pyautogui.press("volumedown")
        print("Volume Down")

# Function to start gesture-based volume control
def start_volume_control():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirrored view
        frame = cv2.flip(frame, 1)
        global frame_height, frame_width
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Call volume control function
                control_volume(hand_landmarks)

        # Display the video feed
        cv2.imshow("Volume Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to listen for voice commands
def listen_for_command():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("Listening for the command 'Detect Volume'...")
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            command = recognizer.recognize_google(audio).lower()
            print(f"Command received: {command}")
            if "detect volume" in command:
                print("Starting gesture-based volume control...")
                start_volume_control()
            else:
                print("Invalid command. Please say 'Detect Volume' to start.")
    except sr.UnknownValueError:
        print("Could not understand the audio.")
    except sr.RequestError:
        print("Could not request results; check your network connection.")

# Run the voice recognition system
if __name__ == "__main__":
    listen_for_command()
