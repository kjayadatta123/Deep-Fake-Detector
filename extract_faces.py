import os
import cv2
from facenet_pytorch import MTCNN
from tqdm import tqdm

mtcnn = MTCNN(keep_all=True)

video_folder = 'videos'
output_folder = 'dataset'
os.makedirs(output_folder, exist_ok=True)

MAX_FRAMES_PER_VIDEO = 30

def extract_faces_from_video(video_path, save_folder, label):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frames = 0

    while cap.isOpened() and saved_frames < MAX_FRAMES_PER_VIDEO:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(frame_rgb)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)

                # Ensure the coordinates are within the frame size
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame_rgb.shape[1], x2)
                y2 = min(frame_rgb.shape[0], y2)

                if x2 - x1 > 0 and y2 - y1 > 0:
                    face = frame_rgb[y1:y2, x1:x2]
                    face = cv2.resize(face, (224, 224))

                    save_path = os.path.join(save_folder, f"{label}_{os.path.basename(video_path).split('.')[0]}_frame{frame_count}.jpg")
                    cv2.imwrite(save_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
                    saved_frames += 1
                    break  # Save only the first detected face per frame

        frame_count += 1

    cap.release()
    print(f"✅ Extracted {saved_frames} faces from {video_path}")

for label in ['REAL', 'FAKE']:
    video_dir = os.path.join(video_folder, label)
    save_dir = os.path.join(output_folder, label)
    os.makedirs(save_dir, exist_ok=True)

    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]

    print(f"\nProcessing {label} videos...")
    for video_file in tqdm(video_files):
        video_path = os.path.join(video_dir, video_file)
        extract_faces_from_video(video_path, save_dir, label)

print("\n✅ Face extraction completed!")
