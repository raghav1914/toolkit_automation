import tkinter as tk
from tkinter import messagebox, filedialog
import webbrowser
import random
import pytz
import requests
import json
import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set up the GUI window
window = tk.Tk()
window.title("Menu")
window.geometry("1920x1080")
from PIL import Image, ImageTk
window.configure(bg="black")

# Load the logo image
logo_image = Image.open("inc.jpeg")
logo_image = logo_image.resize((250, 250))  # Resize the logo image as per your requirement

# Convert the logo image to Tkinter-compatible format
logo_tk = ImageTk.PhotoImage(logo_image)

# Create a Label widget to display the logo
logo_label = tk.Label(window, image=logo_tk)
logo_label.pack()

# Function to display a message box with the given text
def display_message(text):
    messagebox.showinfo("Message", text)

# Function to open a file dialog and return the selected file path
def open_file_dialog():
    file_path = filedialog.askopenfilename()
    return file_path

# Button Functions

def open_website(url):
    webbrowser.open(url)

def send_whatsapp():
    import pywhatkit

    # Define the list of recipients and messages
    recipients = ["+918005757690"]
    messages = ["Hello!", "How are you?", "Sending you a message."]

    # Set the common time for sending messages
    hour = 15
    minute = 59

    # Loop through the recipients and send messages
    for recipient, message in zip(recipients, messages):
        pywhatkit.sendwhatmsg(recipient, message, hour, minute)
        minute += 1  # Increment minute for each subsequent message

        print(f"Message sent to {recipient}: {message}")
    display_message("Sending WhatsApp message")

def send_message():
    from twilio.rest import Client

    account_sid = 'AC279242798af6f7b8eb68bf456fabd420'
    auth_token = 'e64fcd2b58091958cf2e8c31e5dbd165'
    from_number = '+14849928592'
    to_number = '+918005757690'
    message="hihihhi"

    try:
        # Create a Twilio client
        client = Client(account_sid, auth_token)

        # Send the text message
        message = client.messages.create(
            body=message,
            from_=from_number,
            to=to_number
        )

        print("Text message sent successfully!")
        print(f"Message SID: {message.sid}")
    except Exception as e:
        print(f"An error occurred while sending the text message: {str(e)}")

    display_message("Sending message")

def get_time():
    current_time = datetime.datetime.now(pytz.timezone('Asia/Kolkata')).strftime("%I:%M %p")
    print(f"The current time is {current_time}")
    display_message(current_time)

def draw_indian_flag():
    import numpy as np
    import matplotlib.pyplot as py
    import matplotlib.patches as patch

    #Plotting the tri colours in national flag
    a = patch.Rectangle((0, 1), width=12, height=2, facecolor='green', edgecolor='grey')
    b = patch.Rectangle((0, 3), width=12, height=2, facecolor='white', edgecolor='grey')
    c = patch.Rectangle((0, 5), width=12, height=2, facecolor='#FF9933', edgecolor='grey')
    m, n = py.subplots()
    n.add_patch(a)
    n.add_patch(b)
    n.add_patch(c)

    #AshokChakra Circle
    radius = 0.8
    py.plot(6, 4, marker='o', markerfacecolor='#000000ff', markersize=9.5)
    chakra = py.Circle((6, 4), radius, color='#000000ff', fill=False, linewidth=7)
    n.add_artist(chakra)

    #24 spokes in AshokChakra
    for i in range(0, 24):
        p = 6 + radius / 2 * np.cos(np.pi * i / 12 + np.pi / 48)
        q = 6 + radius / 2 * np.cos(np.pi * i / 12 - np.pi / 48)
        r = 4 + radius / 2 * np.sin(np.pi * i / 12 + np.pi / 48)
        s = 4 + radius / 2 * np.sin(np.pi * i / 12 - np.pi / 48)
        t = 6 + radius * np.cos(np.pi * i / 12)
        u = 4 + radius * np.sin(np.pi * i / 12)
        n.add_patch(patch.Polygon([[6, 4], [p, r], [t, u], [q, s]], fill=True, closed=True, color='#000000ff'))
    py.axis('equal')
    py.show()

    display_message("Drawing Indian Flag")

def present_titanic_model():
    # Step 1: Load the dataset
    url = 'https://www.kaggle.com/datasets/yasserh/titanic-dataset'
    df = pd.read_csv('train.csv')

    # Step 2: Data preprocessing
    # Drop irrelevant columns
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Fare', "Embarked"], axis=1)

    # Fill missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)

    # Encode categorical variables
    label_encoder = LabelEncoder()
    df['Sex'] = label_encoder.fit_transform(df['Sex'])

    # Split features and target variable
    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Model training
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Step 6: Predict on test set
    y_pred = model.predict(X_test)

# Step 7: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    display_message("Presenting Titanic Model")

def volume_controller():
    import cv2
    import mediapipe as mp
    from math import hypot
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    import numpy as np
    import pygame

    # Initialize Pygame mixer for audio playback
    pygame.mixer.init()

    # Connect to the default camera
    cap = cv2.VideoCapture(0)

    # Initialize mediapipe hands
    mp_Hands = mp.solutions.hands
    hands = mp_Hands.Hands()
    mp_Draw = mp.solutions.drawing_utils

    # Accessing the speakers using pycaw
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume.iid, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))

    # Find the volume range between the minimum and maximum volume
    volMin, volMax = volume.GetVolumeRange()[:2]

    # Create a Pygame mixer channel for music playback
    music_channel = pygame.mixer.Channel(0)

    # Capturing an image from the camera
    while True:
        status, image = cap.read()
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(imageRGB)

        lmlist = []

        if results.multi_hand_landmarks:
            for handlandmark in results.multi_hand_landmarks:
                for id, lm in enumerate(handlandmark.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmlist.append([id, cx, cy])
                mp_Draw.draw_landmarks(image, handlandmark, mp_Hands.HAND_CONNECTIONS)

        if lmlist != []:
            x1, y1 = lmlist[4][1], lmlist[4][2]
            x2, y2 = lmlist[8][1], lmlist[8][2]

            cv2.circle(image, (x1, y1), 15, (255, 0, 0), cv2.FILLED)
            cv2.circle(image, (x2, y2), 15, (255, 0, 0), cv2.FILLED)
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)

            length = hypot(x2 - x1, y2 - y1)
            vol = np.interp(length, [15, 220], [volMin, volMax])

            volume.SetMasterVolumeLevel(vol, None)

            # Play/pause music when hand distance crosses a threshold
            if length < 50:
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.pause()
                else:
                    pygame.mixer.music.unpause()

        cv2.imshow('Image', image)
        if cv2.waitKey(1) == 27:  # Press ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

    display_message("Opening Volume Controller")

def cursor():
    import cv2
    import mediapipe as mp
    import pyautogui

    cap = cv2.VideoCapture(0)
    hand_detector= mp.solutions.hands.Hands()
    drawing_utils  = mp.solutions.drawing_utils
    screen_width, screen_height = pyautogui.size()
    index_y = 0

    while True:
        _, frame=cap.read()
        frame = cv2.flip(frame,1)
        frame_height, frame_width, _ =frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = hand_detector.process(rgb_frame)
        hands = output.multi_hand_landmarks
        if hands:
            for hand in hands:
                drawing_utils.draw_landmarks(frame, hand)
                landmarks=hand.landmark
                for id, landmark in enumerate(landmarks):   
                    x = int(landmark.x * frame_width)
                    y = int(landmark.y * frame_height)

                    if id == 8:
                        cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255))
                        index_x = screen_width/frame_width*x
                        index_y = screen_height/frame_height*y

                    if id == 4:
                        cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255))
                        thumb_x = screen_width/frame_width*x
                        thumb_y = screen_height/frame_height*y
                        print('outside', abs(index_y-thumb_y))
                        if abs(index_y-thumb_y) < 20:
                            pyautogui.click()
                            pyautogui.sleep(1)
                        elif abs(index_y-thumb_y) < 100:
                            pyautogui.moveTo(index_x, index_y)
        cv2.imshow('Hand Mouse', frame)
        cv2.waitKey(1)

    display_message("Cursor functionality")

def add_sunglasses():
    import cv2

    # Load the cascade for face detection
    face_cascade = cv2.CascadeClassifier('haarcascade_face.xml')

    # Load the bunny face image with an alpha channel
    glasses = cv2.imread('glasses.png', cv2.IMREAD_UNCHANGED)

    # Initialize the video capture
    cap = cv2.VideoCapture(0)

    # Extract the bunny face and the alpha channel
    glasses_image = glasses[:, :, :3]
    glasses_mask = glasses[:, :, 3]

    while True:
        # Read the frame from the video capture
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Process each detected face
        for (x, y, w, h) in faces:
            # Calculate the aspect ratio to resize the bunny face
            aspect_ratio = w / h

            # Resize the bunny face to match the size of the detected face
            resized_glasses = cv2.resize(glasses_image, (int(w * aspect_ratio), h))
            resized_glasses_mask = cv2.resize(glasses_mask, (int(w * aspect_ratio), h))

            # Apply the bunny face mask to create a region of interest (ROI)
            roi = frame[y:y + h, x:x + int(w * aspect_ratio)]
            roi_glasses = cv2.bitwise_and(resized_glasses, resized_glasses, mask=resized_glasses_mask)

            # Add the bunny face to the ROI
            glasses_final = cv2.add(roi, roi_glasses)

            # Update the frame with the bunny face
            frame[y:y + h, x:x + int(w * aspect_ratio)] = glasses_final

        # Display the resulting frame
        cv2.imshow('Sunglasses', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture
    cap.release()

    # Destroy all windows
    cv2.destroyAllWindows()

    display_message("Adding Sunglasses")

def tell_distance():
    import cv2
    import math

    # Load the Haar Cascade classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_face.xml')

    cap = cv2.VideoCapture(0)

    # Constants for distance measurement
    KNOWN_DISTANCE = 100  # Define a known distance (in cm) from the camera to the face
    KNOWN_FACE_WIDTH = 15  # Define the width of the face (in cm) at the known distance

    while True:
        # Read the frame from the webcam
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Process each detected face
        for (x, y, w, h) in faces:
            # Calculate the distance to the face using the known distance and face width
            face_width_pixels = w
            distance = (KNOWN_FACE_WIDTH * cap.get(3)) / (2 * face_width_pixels * math.tan(cap.get(4) * math.pi / 360))

            # Draw the distance on the frame
            cv2.putText(frame, f"Distance: {distance:.2f} cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)

        # Display the frame with distance information
        cv2.imshow('Distance Measurement', frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    display_message("Distance Measurement")

# Create the buttons

btn_send_whatsapp = tk.Button(window, text="Send WhatsApp", command=send_whatsapp,padx=10, pady=5, bd=1, relief=tk.RAISED)
btn_send_whatsapp.pack(padx=10, pady=10)
btn_send_whatsapp.pack()

btn_send_message = tk.Button(window, text="Send Message", command=send_message,padx=10, pady=5, bd=1, relief=tk.RAISED)
btn_send_message.pack(padx=10, pady=10)
btn_send_message.pack()



btn_get_time = tk.Button(window, text="Get Time", command=get_time,padx=10, pady=5, bd=1, relief=tk.RAISED)
btn_get_time.pack(padx=10, pady=10)
btn_get_time.pack()

btn_open_youtube = tk.Button(window, text="Open YouTube", command=lambda: open_website('https://www.youtube.com'),padx=10, pady=5, bd=1, relief=tk.RAISED)
btn_open_youtube.pack()
btn_open_youtube.pack(padx=10, pady=10)

btn_draw_indian_flag = tk.Button(window, text="Draw Indian Flag", command=draw_indian_flag,padx=10, pady=5, bd=1, relief=tk.RAISED)
btn_draw_indian_flag.pack()
btn_draw_indian_flag.pack(padx=10, pady=10)

btn_present_titanic_model = tk.Button(window, text="Present Titanic Model", command=present_titanic_model,padx=10, pady=5, bd=1, relief=tk.RAISED)
btn_present_titanic_model.pack()
btn_present_titanic_model.pack(padx=10, pady=10)

btn_volume_controller = tk.Button(window, text="Volume Controller", command=volume_controller,padx=10, pady=5, bd=1, relief=tk.RAISED)
btn_volume_controller.pack()
btn_volume_controller.pack(padx=10, pady=10)

btn_cursor = tk.Button(window, text="Cursor", command=cursor,padx=10, pady=5, bd=1, relief=tk.RAISED)
btn_cursor.pack()
btn_cursor.pack(padx=10, pady=10)

btn_add_sunglasses = tk.Button(window, text="Add Sunglasses", command=add_sunglasses,padx=10, pady=5, bd=1, relief=tk.RAISED)
btn_add_sunglasses.pack()
btn_add_sunglasses.pack(padx=10, pady=10)

btn_tell_distance = tk.Button(window, text="Tell Distance", command=tell_distance,padx=10, pady=5, bd=1, relief=tk.RAISED)
btn_tell_distance.pack()
btn_tell_distance.pack(padx=10,pady=10)

# Run the Tkinter event loop
window.mainloop()
