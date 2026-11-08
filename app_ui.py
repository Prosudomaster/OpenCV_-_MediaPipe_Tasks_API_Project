import tkinter as tk
import cv2
from PIL import Image, ImageTk
import customtkinter as ctk
from face_processor import FaceMeshProcessor

# UI Theme Settings
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class FaceMeshApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Face Mesh Studio - OpenCV & MediaPipe")
        self.geometry("900x700")
        self.resizable(False, False)

        # Initialize Processor
        self.processor = FaceMeshProcessor()
        self.cap = None
        self.is_running = False

        # --- UI Components ---
        # Title Label
        self.title_label = ctk.CTkLabel(self, text="Real-Time Face Mesh Detection", font=("Arial", 20, "bold"))
        self.title_label.pack(pady=15)

        # Video Frame Container
        self.video_frame = ctk.CTkFrame(self, width=640, height=480, corner_radius=10)
        self.video_frame.pack(pady=10)

        self.image_label = tk.Label(self.video_frame, bg="#1a1a1a")
        self.image_label.pack(fill="both", expand=True)

        # Control Buttons Frame
        self.btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.btn_frame.pack(pady=20)

        self.start_btn = ctk.CTkButton(self.btn_frame, text="Start Camera", command=self.start_camera, fg_color="#2ecc71", hover_color="#27ae60", font=("Arial", 14, "bold"), width=150, height=40)
        self.start_btn.grid(row=0, column=0, padx=15)

        self.stop_btn = ctk.CTkButton(self.btn_frame, text="Stop Camera", command=self.stop_camera, fg_color="#e74c3c", hover_color="#c0392b", font=("Arial", 14, "bold"), width=150, height=40, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=15)

    def start_camera(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            self.is_running = True
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self.update_frame()

    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.image_label.configure(image="")
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")

    def update_frame(self):
        if self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Process frame using our backend processor
                processed_frame = self.processor.process_frame(frame)

                # Convert OpenCV frame (BGR) to PIL Image (RGB)
                cv2image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                img = img.resize((640, 480))
                imgtk = ImageTk.PhotoImage(image=img)

                self.image_label.imgtk = imgtk
                self.image_label.configure(image=imgtk)

            self.after(10, self.update_frame)

    def on_closing(self):
        self.stop_camera()
        self.destroy()

if __name__ == "__main__":
    app = FaceMeshApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()