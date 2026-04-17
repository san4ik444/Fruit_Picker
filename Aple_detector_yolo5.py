"""
🍎 БЫСТРЫЙ ДЕТЕКТОР ЯБЛОК YOLOv5 - ОПТИМИЗИРОВАННАЯ ВЕРСИЯ
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import sys
import time
from datetime import datetime

# ===== НАСТРОЙКИ ДЛЯ СКОРОСТИ (МЕНЯЙТЕ ЗДЕСЬ) =====
FRAME_WIDTH = 640  # Было 640 - уменьшили в 2 раза
FRAME_HEIGHT = 480  # Было 480 - уменьшили в 2 раза
MODEL_SIZE = 320  # Размер для нейросети (было 640)
CONF_THRESHOLD = 0.5  # Порог уверенности
SKIP_FRAMES = 1  # 0=все кадры, 1=каждый второй, 2=каждый третий
USE_NANO_MODEL = True  # True = yolov5n.pt (быстрее), False = yolov5s.pt
# ==================================================

# Очищаем путь
sys.path = [p for p in sys.path if 'models' not in p.lower()]

# Добавляем YOLOv5
yolo_path = Path(__file__).parent / 'yolov5'
if not yolo_path.exists():
    print(f"[ERROR] YOLOv5 not found at {yolo_path}")
    sys.exit(1)

sys.path.insert(0, str(yolo_path))

# Импорты
try:
    from models.common import DetectMultiBackend
    from utils.general import non_max_suppression, scale_boxes
    from utils.augmentations import letterbox
    from utils.torch_utils import select_device

    print("[INFO] YOLOv5 modules imported")
except ImportError as e:
    print(f"[ERROR] Cannot import: {e}")
    sys.exit(1)


class FastAppleDetector:
    def __init__(self):
        # Выбираем модель
        if USE_NANO_MODEL:
            weights_path = str(yolo_path / 'yolov5n.pt')
        else:
            weights_path = str(yolo_path / 'yolov5s.pt')

        self.device = select_device('cpu')
        self.conf_thres = CONF_THRESHOLD

        print(f"[INFO] Loading model: {weights_path}")

        # Проверяем и скачиваем модель
        if not Path(weights_path).exists():
            print("[INFO] Downloading model...")
            import urllib.request
            if USE_NANO_MODEL:
                url = "https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n.pt"
            else:
                url = "https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt"
            urllib.request.urlretrieve(url, weights_path)
            print("[INFO] Model downloaded!")

        # Загружаем модель с меньшим размером изображения
        self.model = DetectMultiBackend(weights_path, device=self.device)
        self.stride = self.model.stride
        self.img_size = (MODEL_SIZE, MODEL_SIZE)  # Уменьшенный размер!
        self.apple_class_id = 47

        print(f"[INFO] Model ready! Image size: {MODEL_SIZE}x{MODEL_SIZE}")

    def preprocess(self, frame):
        """Быстрая подготовка кадра"""
        img = letterbox(frame, self.img_size, stride=self.stride, auto=True)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def detect(self, frame):
        """Быстрое обнаружение яблок"""
        original_shape = frame.shape
        img = self.preprocess(frame)
        pred = self.model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, self.conf_thres, 0.45)

        apples = []
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], original_shape).round()
                for *xyxy, conf, cls in reversed(det):
                    if int(cls) == self.apple_class_id:
                        x1, y1, x2, y2 = map(int, xyxy)
                        apples.append({'bbox': (x1, y1, x2, y2), 'confidence': float(conf)})
        return apples

    def draw_detections(self, frame, apples):
        """Рисует рамки"""
        for apple in apples:
            x1, y1, x2, y2 = apple['bbox']
            conf = apple['confidence']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"APPLE", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame


def main():
    print("=" * 60)
    print("🚀 БЫСТРЫЙ ДЕТЕКТОР ЯБЛОК YOLOv5")
    print("=" * 60)
    print(f"Размер кадра: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"Размер модели: {MODEL_SIZE}x{MODEL_SIZE}")
    print(f"Обработка кадров: каждый {SKIP_FRAMES + 1}-й")
    print("=" * 60)

    # Проверяем YOLOv5
    yolo_path = Path(__file__).parent / 'yolov5'
    if not yolo_path.exists():
        print("[ERROR] YOLOv5 folder not found!")
        return

    # Создаем детектор
    try:
        detector = FastAppleDetector()
    except Exception as e:
        print(f"[ERROR] Failed: {e}")
        return

    # Открываем камеру с МАЛЕНЬКИМ разрешением
    print("[INFO] Opening camera...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("[ERROR] Cannot open camera!")
            return

    # Устанавливаем маленькое разрешение для быстрой работы
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("[INFO] Camera ready!")
    print("[INFO] Controls: Q - Quit, S - Save")
    print("=" * 60)

    frame_count = 0
    total_apples = 0
    start_time = time.time()

    # Для пропуска кадров
    last_apples = []
    process_counter = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame = cv2.flip(frame, 1)

            # Детекция НЕ на каждом кадре (экономия ресурсов)
            if frame_count % (SKIP_FRAMES + 1) == 0:
                process_counter += 1
                last_apples = detector.detect(frame)
                total_apples += len(last_apples)

            # Отрисовка (быстрая)
            result = detector.draw_detections(frame.copy(), last_apples)

            # Статистика
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            # Информация на экране
            cv2.putText(result, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(result, f"Apples: {len(last_apples)}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(result, f"Total: {total_apples}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(result, "Q-Quit | S-Save", (10, result.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Fast Apple Detector", result)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"apple_{timestamp}.jpg"
                cv2.imwrite(filename, result)
                print(f"[INFO] Saved: {filename}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("FINAL STATISTICS")
        print("=" * 60)
        print(f"Time: {elapsed:.1f} seconds")
        print(f"Frames: {frame_count}")
        print(f"Processed frames: {process_counter}")
        if elapsed > 0:
            print(f"Display FPS: {frame_count / elapsed:.1f}")
        print(f"Total apples: {total_apples}")
        print("=" * 60)


if __name__ == "__main__":
    main()