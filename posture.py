import os
import urllib.request

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarksConnections,
    drawing_utils,
    RunningMode,
)

# Pose landmarker model (lite) - download once if missing
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_lite.task")


def _ensure_model():
    if not os.path.isfile(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        print("Downloading pose landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")
    return MODEL_PATH


def main():
    model_path = _ensure_model()

    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
    options = PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.VIDEO,
    )
    landmarker = PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    frame_index = 0

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))

            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC)) if cap.get(cv2.CAP_PROP_POS_MSEC) else frame_index * 33
            frame_index += 1

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                for pose_landmarks in result.pose_landmarks:
                    drawing_utils.draw_landmarks(
                        frame,
                        pose_landmarks,
                        connections=PoseLandmarksConnections.POSE_LANDMARKS,
                    )

            cv2.imshow("Pose Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
