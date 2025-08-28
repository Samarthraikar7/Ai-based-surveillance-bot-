import cv2
import os
import sys
import torch
import pytesseract
import face_recognition
import pyttsx3
import socket
import pickle
import struct
from PIL import Image
from datetime import datetime
from pushbullet import Pushbullet
from transformers import BlipProcessor, BlipForConditionalGeneration

# Tesseract config (keep your local path)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Pushbullet setup
pb = Pushbullet("xyz")  # Replace with your token

# Text-to-speech
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Load BLIP model
def load_blip_model():
    print("Loading BLIP model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    print("Model loaded.")
    return processor, model

# Get video stream from Raspberry Pi
def get_pi_video_stream():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(('123.123.123', 1234))  # 🔁 Replace with your Pi's IP address

    data = b""
    payload_size = struct.calcsize(">I")  # Match with Pi: big-endian unsigned int

    try:
        while True:
            # Get message size
            while len(data) < payload_size:
                packet = client_socket.recv(4096)
                if not packet:
                    return
                data += packet

            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack(">I", packed_msg_size)[0]

            # Get full frame
            while len(data) < msg_size:
                packet = client_socket.recv(4096)
                if not packet:
                    return
                data += packet

            frame_data = data[:msg_size]
            data = data[msg_size:]

            try:
                frame = pickle.loads(frame_data)
                yield frame
            except Exception as e:
                print("⚠️ Failed to decode frame:", e)
                continue

    finally:
        client_socket.close()

# Modified capture function to use Pi's webcam
def capture_image(filename="captured.jpg"):
    print("Press 's' to capture or 'q' to quit.")
    for frame in get_pi_video_stream():
        cv2.imshow("Capture from Pi", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            cv2.imwrite(filename, frame)
            print("Saved to", filename)
            break
        elif key == ord('q'):
            cv2.destroyAllWindows()
            sys.exit()
    cv2.destroyAllWindows()

# Face recognition (unchanged)
def recognize_faces(image_path, known_face_encodings, known_face_names):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            name = known_face_names[matches.index(True)]
        face_names.append(name)
    return face_names

# BLIP caption (unchanged)
def generate_caption(image_path, processor, model):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# OCR (unchanged)
def extract_text(image_path):
    image = Image.open(image_path)
    return pytesseract.image_to_string(image)

# Modified describe_scene to use Pi's webcam
def describe_scene():
    processor, model = load_blip_model()
    known_faces_dir = r"D:\vvscode\opencv\known_faces"  # Keep your local path
    known_face_encodings = []
    known_face_names = []

    # Load known faces (unchanged)
    for file in os.listdir(known_faces_dir):
        if file.endswith(".jpg") or file.endswith(".png"):
            path = os.path.join(known_faces_dir, file)
            image = face_recognition.load_image_file(path)
            enc = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(enc)
            known_face_names.append(os.path.splitext(file)[0])

    capture_image("captured.jpg")
    face_names = recognize_faces("captured.jpg", known_face_encodings, known_face_names)

    for name in face_names:
        if name != "Unknown":
            print(f"Recognized: {name}")
            speak_text(f"Hello {name}")
            pb.push_note("Face Recognition", f"{name} recognized.")
        else:
            print("Unknown person.")
            speak_text("Unknown person detected")
            pb.push_note("Face Recognition", "Unknown person detected.")

    caption = generate_caption("captured.jpg", processor, model)
    print("Caption:", caption)
    text = extract_text("captured.jpg").strip()
    print("Text:", text if text else "No text detected.")

    result = f"{caption}. The text says: {text}" if text else caption
    speak_text(result)
    pb.push_note("Scene Description", result)

# Modified detect_humans to use Pi's webcam
def detect_humans():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    alert_sent = False

    for frame in get_pi_video_stream():
        boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

        if len(boxes) > 0 and not alert_sent:
            description = f"👤 {len(boxes)} human(s) detected\n"
            for i, (x, y, w, h) in enumerate(boxes):
                center_x = x + w // 2
                position = "left" if center_x < frame.shape[1] // 3 else "right" if center_x > 2 * frame.shape[1] // 3 else "center"
                size = "near" if h > 300 else "far"
                description += f" - Person {i+1}: Position={position}, Size={size}\n"
            description += f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            pb.push_note("🚨 Human Detection", description)
            print(description)
            alert_sent = True
        elif len(boxes) == 0:
            alert_sent = False

        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Human Detection from Pi", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

# Modified live_feed to use Pi's webcam
def live_feed():
    print("Press 'q' to quit live view.")
    for frame in get_pi_video_stream():
        cv2.imshow("Live Feed from Pi", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

# CLI main (unchanged)
def main():
    print("\nChoose a mode:")
    print("1. describe - Capture and analyze scene")
    print("2. detect   - Live human detection")
    print("3. live     - Show webcam live feed")
    cmd = input("Enter command: ").strip().lower()

    if cmd == "1":
        describe_scene()
    elif cmd == "2":
        detect_humans()
    elif cmd == "3":
        live_feed()
    else:
        print("❌ Unknown command.")

if __name__ == "__main__":

     main()
