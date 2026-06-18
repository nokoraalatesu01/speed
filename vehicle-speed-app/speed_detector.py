import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import os


# =====================================================
# CONFIG
# =====================================================

MODEL_NAME = "yolov8s.pt"

# Ước lượng cho camera I-5 MP 152.3
LANE_WIDTH_METERS = 3.7
NUM_LANES = 5

ROAD_WIDTH_METERS = LANE_WIDTH_METERS * NUM_LANES

WARP_WIDTH = 500
WARP_HEIGHT = 500

SMOOTHING_WINDOW = 10

# Giới hạn tốc độ để loại bỏ nhiễu
MIN_SPEED = 0
MAX_SPEED = 150

# =====================================================
# COLOR UTILS
# =====================================================

def get_track_color(track_id):

    np.random.seed(track_id)

    return (
        int(np.random.randint(50, 255)),  # B
        int(np.random.randint(50, 255)),  # G
        int(np.random.randint(50, 255))   # R
    )

# =====================================================
# HOMOGRAPHY
# =====================================================

def build_homography():

    # Điều chỉnh nếu cần

    src = np.float32([
        [105, 238],   # bottom left
        [315, 238],   # bottom right
        [215, 120],   # top left
        [255, 120]    # top right
    ])

    dst = np.float32([
        [0, WARP_HEIGHT],
        [WARP_WIDTH, WARP_HEIGHT],
        [0, 0],
        [WARP_WIDTH, 0]
    ])

    return cv2.getPerspectiveTransform(src, dst)


# =====================================================
# SPEED ESTIMATOR
# =====================================================

class SpeedEstimator:

    def __init__(self, fps):

        self.fps = fps

        self.track_history = defaultdict(
            lambda: deque(maxlen=15)
        )

        self.speed_history = defaultdict(
            lambda: deque(maxlen=SMOOTHING_WINDOW)
        )

        self.H = build_homography()

        self.meters_per_pixel = (
            ROAD_WIDTH_METERS /
            WARP_WIDTH
        )

    def transform_point(self, x, y):

        point = np.array(
            [[[x, y]]],
            dtype=np.float32
        )

        transformed = cv2.perspectiveTransform(
            point,
            self.H
        )

        return transformed[0][0]

    def estimate(self, track_id, x, y, frame_idx):

        bx, by = self.transform_point(x, y)

        self.track_history[track_id].append(
            (bx, by, frame_idx)
        )

        history = self.track_history[track_id]

        if len(history) < 5:
            return 0

        old_x, old_y, old_frame = history[0]

        dist_pixels = np.sqrt(
            (bx - old_x) ** 2 +
            (by - old_y) ** 2
        )

        dist_meters = (
            dist_pixels *
            self.meters_per_pixel
        )

        dt = (
            frame_idx - old_frame
        ) / self.fps

        if dt <= 0:
            return 0

        speed_mps = dist_meters / dt

        speed_kmh = speed_mps * 3.6

        if speed_kmh > MAX_SPEED:
            speed_kmh = MAX_SPEED

        if speed_kmh < MIN_SPEED:
            speed_kmh = MIN_SPEED

        self.speed_history[track_id].append(
            speed_kmh
        )

        return np.mean(
            self.speed_history[track_id]
        )


# =====================================================
# DRAWING
# =====================================================

def draw_vehicle(
    frame,
    box,
    speed,
    track_id
):

    color = get_track_color(track_id)

    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(
        frame,
        (x1,y1),
        (x2,y2),
        color,
        3
    )

    cv2.putText(
        frame,
        f"ID {track_id}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2
    )


# =====================================================
# VIDEO PROCESSOR
# =====================================================

def process_video(
        video_path,
        output_path="result.mp4"
):

    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    print("Loading model...")

    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print("FPS:", fps)
    print("Frames:", total_frames)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    speed_estimator = SpeedEstimator(
        fps
    )

    frame_idx = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_idx += 1
        speed_board = []

        # README: frame đầu bị lỗi
        if frame_idx == 1:
            continue

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            classes=[
                2,  # car
                3,  # motorcycle
                5,  # bus
                7   # truck
            ]
        )

        if (
            len(results) == 0 or
            results[0].boxes.id is None
        ):

            writer.write(frame)
            continue

        boxes = (
            results[0]
            .boxes
            .xyxy
            .cpu()
            .numpy()
        )

        ids = (
            results[0]
            .boxes
            .id
            .cpu()
            .numpy()
            .astype(int)
        )

        for box, track_id in zip(
                boxes,
                ids):
            x1, y1, x2, y2 = box

            bottom_center_x = (
                x1 + x2
            ) / 2

            bottom_center_y = y2

            speed = speed_estimator.estimate(
                track_id,
                bottom_center_x,
                bottom_center_y,
                frame_idx
            )

            draw_vehicle(
                frame,
                box,
                speed,
                track_id
            )

            speed_board.append(
                (
                    track_id,
                    speed
                )
            )

# =====================================================
# SPEED BOARD
# =====================================================

    panel_width = 250

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (width - panel_width, 0),
        (width, height),
        (0, 0, 0),
        -1
    )

    frame = cv2.addWeighted(
        overlay,
        0.35,
        frame,
        0.65,
        0
    )

    cv2.putText(
        frame,
        "SPEED",
        (width - panel_width + 20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    speed_board = sorted(
        speed_board,
        key=lambda x: x[1],
        reverse=True
    )

    for idx, (track_id, speed) in enumerate(
            speed_board[:10]):

        color = get_track_color(track_id)

        y = 70 + idx * 35

        # ô màu
        cv2.rectangle(
            frame,
            (width - panel_width + 15, y - 12),
            (width - panel_width + 40, y + 12),
            color,
            -1
        )

        cv2.putText(
            frame,
            f"ID {track_id}",
            (width - panel_width + 50, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

        cv2.putText(
            frame,
            f"{int(speed)} km/h",
            (width - 120, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2
        )

        writer.write(frame)



        if frame_idx % 20 == 0:

            print(
                f"{frame_idx}/{total_frames}"
            )
    

    cap.release()
    writer.release()

    print(
        "Saved:",
        output_path
    )

    return output_path


# =====================================================
# LOCAL TEST
# =====================================================

if __name__ == "__main__":

    INPUT_VIDEO = "test.avi"

    process_video(
        INPUT_VIDEO,
        "output.mp4"
    )