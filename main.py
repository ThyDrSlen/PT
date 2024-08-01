import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

class PT4UApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PT4U - Pose Detection")
        self.root.geometry("1920x1080")
        self.root.config(background="#000000")

        #User Interface Customizations 
        button_style = {
            "font": ("Helvetica", 20, "bold"),
            "bg": "#303030",
            "fg": "#000000",
            "activebackground": "#00FF00",
            "width": 12,
            "height": 3,
            "bd": 5
        }

        self.camera_button = tk.Button(root, text="Start Camera", command=self.start_camera, **button_style)
        self.upload_button = tk.Button(root, text="Upload Video", command=self.upload_video, **button_style)
        self.pause_button = tk.Button(root, text="Pause", command=self.toggle_pause, **button_style)

        #Playback Slider 
        self.speed_slider = tk.Scale(root, from_=0.3, to=3.0, resolution=0.1, orient="horizontal", label="Playback Speed", font=("Helvetica", 12, "bold"), length=275)
        self.speed_slider.set(1.0)  # Set the default speed to 1.0 (normal speed)

        # Arrange buttons and slider in the window
        self.video_label = tk.Label(root)
        self.video_label.pack(pady=20)

        self.camera_button.pack(side=tk.LEFT, padx=20, pady=20)
        self.upload_button.pack(side=tk.LEFT, padx=20, pady=20)
        self.pause_button.pack(side=tk.LEFT, padx=20, pady=20)
        self.speed_slider.pack(side=tk.LEFT, padx=20, pady=20)

        self.cap = None
        self.delay = 15  # Default delay
        self.paused = False  # Variable to keep track of pause state

    
    #TODO fix check math on this 
    #90 degrees is not showing up as a correct right angle its more like 93 degrees right now maybe more
    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360.0 - angle

        return angle

    
    #fix this function name 
    
    def draw_angle_with_outline(self, frame, angle, position, color):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 2
        thickness = 3
        color_black = (0, 0, 0)
        if not isinstance(position, tuple):
            position = tuple(position)
        cv2.putText(frame, str(int(angle)), position, font, scale, color_black, thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, str(int(angle)), position, font, scale, color, thickness, cv2.LINE_AA)

    def process_pose_landmarks(self, pose_results, frame):
        if not pose_results.pose_landmarks:
            return
        landmarks = pose_results.pose_landmarks.landmark

        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

        left_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)

        left_color = self.get_color(left_angle)
        right_color = self.get_color(right_angle)

        self.draw_angle_with_outline(frame, left_angle, tuple(np.multiply(left_elbow, [frame.shape[1], frame.shape[0]]).astype(int)), left_color)
        self.draw_angle_with_outline(frame, right_angle, tuple(np.multiply(right_elbow, [frame.shape[1], frame.shape[0]]).astype(int)), right_color)

        left_landmarks = [mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value]
        for i in range(len(left_landmarks) - 1):
            start_point = [landmarks[left_landmarks[i]].x, landmarks[left_landmarks[i]].y]
            end_point = [landmarks[left_landmarks[i + 1]].x, landmarks[left_landmarks[i + 1]].y]
            start_point = tuple(np.multiply(start_point, [frame.shape[1], frame.shape[0]]).astype(int))
            end_point = tuple(np.multiply(end_point, [frame.shape[1], frame.shape[0]]).astype(int))
            cv2.line(frame, start_point, end_point, left_color, 2)

        right_landmarks = [mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_WRIST.value]
        for i in range(len(right_landmarks) - 1):
            start_point = [landmarks[right_landmarks[i]].x, landmarks[right_landmarks[i]].y]
            end_point = [landmarks[right_landmarks[i + 1]].x, landmarks[right_landmarks[i + 1]].y]
            start_point = tuple(np.multiply(start_point, [frame.shape[1], frame.shape[0]]).astype(int))
            end_point = tuple(np.multiply(end_point, [frame.shape[1], frame.shape[0]]).astype(int))
            cv2.line(frame, start_point, end_point, right_color, 2)

    def get_color(self, angle):
        deviation = abs(angle - 90)
        normalized_deviation = min(deviation / 15, 1)
        green = int((1 - normalized_deviation) * 255)
        red = int(normalized_deviation * 255)
        blue = 0
        return (red, green, blue)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.paused = False
        self.update_frame()

    def upload_video(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        
        self.cap = cv2.VideoCapture(file_path)
        self.paused = False
        self.update_frame()

    def update_frame(self):
        if self.cap is not None and self.cap.isOpened() and not self.paused:
            ret, frame = self.cap.read()
            if not ret:
                if self.cap.get(cv2.CAP_PROP_FRAME_COUNT) == self.cap.get(cv2.CAP_PROP_POS_FRAMES):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.root.after(self.delay, self.update_frame)
                return

            frame = cv2.resize(frame, (1280, 720))  
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(rgb_frame)
            self.process_pose_landmarks(pose_results, frame)

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            speed = self.speed_slider.get()  
            self.delay = int(1000 / (speed * 30)) 

            self.root.after(self.delay, self.update_frame)

    def toggle_pause(self):
        self.paused = not self.paused
        if not self.paused:
            self.update_frame()

    def on_closing(self):
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()

# Create the main window
root = tk.Tk()
app = PT4UApp(root)
root.protocol("WM_DELETE_WINDOW", app.on_closing)
root.mainloop()
