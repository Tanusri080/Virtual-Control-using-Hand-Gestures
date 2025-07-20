import cv2
import mediapipe as mp
import pyautogui
import time
import speech_recognition as sr

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Function to detect swipe gesture (moving fingers upward or downward)
def detect_swipe(hand_landmarks, prev_index_y):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    # Calculate vertical movement (y-coordinate difference)
    index_y = index_tip.y
    y_movement = prev_index_y - index_y  # Difference in y position from previous frame

    # Adjust movement sensitivity to make scrolling faster
    scroll_sensitivity = 0.03  # Decrease this value for quicker movement

    # Detect swipe up or down (based on the change in y-coordinate)
    if y_movement > scroll_sensitivity:  # Swiping up
        return "up", index_y
    elif y_movement < -scroll_sensitivity:  # Swiping down
        return "down", index_y

    return None, prev_index_y

# Function to start scrolling with gesture detection
def start_scrolling():
    cap = cv2.VideoCapture(0)
    prev_index_y = 0  # Track the previous position of the index finger

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        # Detect hand gestures
        if result.multi_hand_landmarks:
            for hand_landmark in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

                # Gesture Detection for swipe
                swipe_direction, prev_index_y = detect_swipe(hand_landmark, prev_index_y)

                # If swipe is detected, perform scrolling
                if swipe_direction:
                    print(f"Swipe detected: {swipe_direction}")
                    if swipe_direction == "up":
                        pyautogui.scroll(30)  # Scroll up (increase for faster scrolling)
                    elif swipe_direction == "down":
                        pyautogui.scroll(-30)  # Scroll down (increase for faster scrolling)

        # Display the video feed with landmarks
        cv2.imshow("Swipe Gesture Control", frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to listen for the "Scroll" voice command
def listen_for_command():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("Listening for the command 'Scroll'...")
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            command = recognizer.recognize_google(audio).lower()
            print(f"Command received: {command}")
            if "scroll" in command:
                print("Starting swipe gesture-based scrolling...")
                start_scrolling()
            else:
                print("Invalid command. Please say 'Scroll' to start.")
    except sr.UnknownValueError:
        print("Could not understand the audio.")
    except sr.RequestError:
        print("Could not request results; check your network connection.")

# Run the voice recognition system
if __name__ == "__main__":
    listen_for_command()