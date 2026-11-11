"""Обработка кадров и хранение локального профиля."""

from collections import deque
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from threading import Event
from time import monotonic

import cv2
import numpy as np


MODELS = Path(__file__).resolve().parent / "models"
PROFILE = Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".local" / "share") / (
    "RnxMyFaceRecognitor/sface-profile.npy"
)
MATCH_THRESHOLD = 0.48
FRAME_TIMEOUT = 2.0


def normalize(feature):
    length = np.linalg.norm(feature)
    if not np.isfinite(length) or length <= 0:
        raise ValueError("Некорректный вектор лица")
    return feature / length


def load_profile(path):
    try:
        with path.open("rb") as file:
            feature = np.load(file, allow_pickle=False)
    except FileNotFoundError:
        return None
    if not isinstance(feature, np.ndarray) or feature.shape != (128,):
        raise ValueError("Некорректный формат профиля")
    if feature.dtype != np.float32:
        raise ValueError("Некорректный тип данных профиля")
    return normalize(feature)


def save_profile(path, feature):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
    try:
        with temporary.open("wb") as file:
            np.save(file, feature, allow_pickle=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class FaceEngine:
    def __init__(self):
        # Буфер позволяет загрузить модели и из путей с кириллицей в Windows.
        detector = (MODELS / "face_detection_yunet_2023mar.onnx").read_bytes()
        recognizer = (MODELS / "face_recognition_sface_2021dec.onnx").read_bytes()
        self.detector = cv2.FaceDetectorYN.create("onnx", detector, b"", (320, 320), 0.85)
        self.recognizer = cv2.FaceRecognizerSF.create("onnx", recognizer, b"")
        self.input_size = (320, 320)

    def analyze(self, image):
        height, width = image.shape[:2]
        if self.input_size != (width, height):
            self.input_size = (width, height)
            self.detector.setInputSize(self.input_size)
        _, faces = self.detector.detect(image)
        if faces is None:
            return 0, None
        if len(faces) != 1:
            return len(faces), None
        face = faces[0]
        x, y, w, h = face[:4]
        if min(w, h) < min(width, height) * 0.15 or (
            x < 0 or y < 0 or x + w > width or y + h > height
        ):
            return 1, None
        aligned = self.recognizer.alignCrop(image, face)
        feature = self.recognizer.feature(aligned).reshape(-1)
        return 1, normalize(feature)


@dataclass(frozen=True)
class Frame:
    captured_at: float
    image: np.ndarray
    faces: int
    feature: np.ndarray | None


class Camera:
    def __init__(self, index):
        self.index = index
        self.stop = Event()
        self.updates = deque(maxlen=1)

    def open(self):
        backends = (cv2.CAP_DSHOW, cv2.CAP_MSMF) if sys.platform == "win32" else (cv2.CAP_ANY,)
        for backend in backends:
            if self.stop.is_set():
                break
            capture = cv2.VideoCapture(self.index, backend)
            try:
                if capture.isOpened():
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    for _ in range(8):
                        ok, image = capture.read()
                        if ok and image is not None and image.size and not self.stop.is_set():
                            return capture, image
                        if self.stop.wait(0.1):
                            break
            except cv2.error:
                logging.exception("Не удалось открыть камеру через %s", backend)
            capture.release()
        return None, None

    def run(self):
        try:
            engine = FaceEngine()
        except (OSError, cv2.error):
            logging.exception("Не удалось загрузить модели")
            self.updates.append("Проверьте наличие файлов моделей в папке models.")
            return

        capture = None
        try:
            capture, image = self.open()
            if capture is None:
                self.updates.append("Проверьте подключение камеры и разрешение на доступ в Windows.")
                return
            last_frame = monotonic()
            while not self.stop.is_set():
                now = monotonic()
                if image is not None and image.size:
                    last_frame = now
                    height, width = image.shape[:2]
                    scale = min(1.0, 640 / max(width, height))
                    if scale < 1:
                        size = (max(1, round(width * scale)), max(1, round(height * scale)))
                        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
                    count, feature = engine.analyze(image)
                    self.updates.append(Frame(now, image, count, feature))
                    if self.stop.wait(max(0, 0.07 - (monotonic() - now))):
                        break
                elif now - last_frame >= FRAME_TIMEOUT:
                    self.updates.append("Камера перестала передавать изображение.")
                    return
                elif self.stop.wait(0.05):
                    break
                ok, image = capture.read()
                if not ok:
                    image = None
        except (cv2.error, ValueError):
            logging.exception("Не удалось обработать кадр")
            self.updates.append("Не удалось получить или обработать кадр с камеры.")
        finally:
            if capture is not None:
                capture.release()


class Enrollment:
    SAMPLE_COUNT = 5
    SAMPLE_INTERVAL = 0.45
    DURATION = 18

    def __init__(self, started_at):
        self.started_at = started_at
        self.samples = []
        self.last_sample = started_at

    def add(self, frame):
        if frame.captured_at <= self.started_at:
            return None
        if frame.feature is None:
            self.samples.clear()
            return None
        if frame.captured_at - self.last_sample < self.SAMPLE_INTERVAL:
            return None
        if self.samples and np.dot(self.samples[0], frame.feature) < MATCH_THRESHOLD:
            self.samples.clear()
        self.samples.append(frame.feature)
        self.last_sample = frame.captured_at
        if len(self.samples) == self.SAMPLE_COUNT:
            return normalize(np.mean(self.samples, axis=0))
        return None
