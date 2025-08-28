# Raspberry Pi Code to send webcam feed

import cv2
import socket
import struct
import pickle
import time

class PiVideoServer:
    def __init__(self, host='0.0.0.0', port=8485):  
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((host, port))
        self.server_socket.listen(1)
        print(f"[PI] Server started at {host}:{port}")

    def start(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv2.CAP_PROP_FPS, 15)

        try:
            while True:
                client_socket, addr = self.server_socket.accept()
                print(f"[PI] Client connected: {addr}")

                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        data = pickle.dumps(frame, protocol=pickle.HIGHEST_PROTOCOL)
                        message = struct.pack(">I", len(data)) + data  # Use ">I" for consistent size
                        client_socket.sendall(message)
                        time.sleep(0.1)  # ~30 FPS

                except (ConnectionResetError, BrokenPipeError):
                    print("[PI] Client disconnected")

                finally:
                    client_socket.close()

        except KeyboardInterrupt:
            print("[PI] Shutting down...")

        finally:
            cap.release()
            self.server_socket.close()

if __name__ == "__main__":   
    server = PiVideoServer()
    server.start()

