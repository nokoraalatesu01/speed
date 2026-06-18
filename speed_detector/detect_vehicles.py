import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
CLASS_NAMES_VI = {
    2: "car",
    3: "motorbike",
    5: "bus",
    7: "truck",
}
LINE_NAMES = ("A", "B")
DEFAULT_LINE_A = "0.20,0.67,0.58,0.67"
DEFAULT_LINE_B = "0.00,0.81,0.62,0.81"
DEFAULT_CONFIG = {
    "line_a": DEFAULT_LINE_A,
    "line_b": DEFAULT_LINE_B,
    "distance_meters": 10.0,
    "speed_limit_kmh": 60.0,
    "max_speed_kmh": 150.0,
    "confidence": 0.25,
    "tracker": "bytetrack.yaml",
    "detect_roi_only": False,
}


def parse_source(source: str):
    if source.isdigit():
        return int(source)

    source_path = Path(source)
    if source_path.exists():
        return str(source_path)

    parent_source_path = Path("..") / source
    if parent_source_path.exists():
        return str(parent_source_path)

    return source


def get_video_files(source: str):
    parsed_source = parse_source(source)
    if isinstance(parsed_source, int):
        return [parsed_source]

    source_path = Path(parsed_source)
    if source_path.is_dir():
        return sorted(
            path for path in source_path.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS
        )

    return [source_path]


def parse_line_spec(line_spec: str):
    values = [float(value.strip()) for value in line_spec.split(",")]
    if len(values) != 4:
        raise ValueError("Line spec must have 4 comma-separated values: x1,y1,x2,y2.")

    if any(value < 0 or value > 1 for value in values):
        raise ValueError("Line spec values must be ratios between 0 and 1.")

    return tuple(values)


def load_config(config_path: str):
    config = DEFAULT_CONFIG.copy()
    if not config_path:
        return config

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as config_file:
        user_config = json.load(config_file)

    for key in user_config:
        if key not in config:
            raise ValueError(f"Unknown config key: {key}")

    config.update(user_config)
    return config


def save_config(config_path: str, config):
    if not config_path:
        return

    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2)
        config_file.write("\n")


def scale_line(line_spec, width: int, height: int):
    x1_ratio, y1_ratio, x2_ratio, y2_ratio = line_spec
    return (
        (int(width * x1_ratio), int(height * y1_ratio)),
        (int(width * x2_ratio), int(height * y2_ratio)),
    )


def make_roi_polygon(line_a, line_b):
    return np.array([line_a[0], line_a[1], line_b[1], line_b[0]], dtype=np.int32)


def open_video_writer(output_path: str, fps: float, frame_size: tuple[int, int]):
    if not output_path:
        return None

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)


def draw_vehicle(
    frame,
    box,
    class_id: int,
    confidence: float,
    track_id: int | None,
    speed_kmh: float | None = None,
    is_overspeed: bool = False,
):
    x1, y1, x2, y2 = map(int, box)
    vehicle_name = CLASS_NAMES_VI.get(class_id, class_id)
    if track_id is not None:
        label = f"ID {track_id} {vehicle_name} {confidence:.2f}"
    else:
        label = f"{vehicle_name} {confidence:.2f}"

    if speed_kmh is not None:
        label = f"{label} {speed_kmh:.1f} km/h"
        if is_overspeed:
            label = f"{label} OVER"

    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    label_width = label_size[0] + 12
    color = (0, 0, 255) if is_overspeed else (0, 220, 80)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(frame, (x1, max(0, y1 - 26)), (x1 + label_width, y1), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 6, y1 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def draw_track_trail(frame, points):
    if len(points) < 2:
        return

    for point_index in range(1, len(points)):
        cv2.line(frame, points[point_index - 1], points[point_index], (0, 180, 255), 2)


def draw_speed_lines(frame, line_a, line_b):
    roi_polygon = make_roi_polygon(line_a, line_b)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [roi_polygon], (0, 180, 120))
    cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

    lines = (("A", line_a, (255, 80, 255)), ("B", line_b, (0, 220, 255)))

    cv2.line(frame, line_a[0], line_b[0], (255, 255, 255), 4)
    cv2.line(frame, line_a[1], line_b[1], (255, 255, 255), 4)

    for name, line, color in lines:
        start, end = line
        cv2.line(frame, start, end, color, 3)
        label_x = min(start[0], end[0]) + 12
        label_y = min(start[1], end[1]) - 10
        label_y = max(26, label_y)
        cv2.rectangle(frame, (label_x - 8, label_y - 22), (label_x + 84, label_y + 4), color, -1)
        cv2.putText(
            frame,
            f"Line {name}",
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def apply_roi_mask(frame, roi_polygon):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [roi_polygon], 255)
    return cv2.bitwise_and(frame, frame, mask=mask)


def point_inside_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def orientation(a, b, c):
    return (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])


def on_segment(a, b, c):
    return (
        min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
        and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
    )


def segments_intersect(p1, q1, p2, q2):
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 * o2 < 0 and o3 * o4 < 0:
        return True

    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False


def crossed_line_segment(previous_point, current_point, line):
    if previous_point is None or previous_point == current_point:
        return False

    return segments_intersect(previous_point, current_point, line[0], line[1])


def mark_line_crossings(track_id: int, previous_point, current_point, speed_lines, crossed_lines, time_seconds: float):
    events = []
    for line_name, line in zip(LINE_NAMES, speed_lines):
        if line_name in crossed_lines[track_id]:
            continue

        if crossed_line_segment(previous_point, current_point, line):
            crossed_lines[track_id][line_name] = time_seconds
            events.append((track_id, line_name, time_seconds))

    return events


def calculate_speed_kmh(crossing_times, distance_meters: float, max_speed_kmh: float):
    if "A" not in crossing_times or "B" not in crossing_times:
        return None

    elapsed_seconds = abs(crossing_times["B"] - crossing_times["A"])
    if elapsed_seconds <= 0:
        return None

    speed_kmh = distance_meters / elapsed_seconds * 3.6
    if max_speed_kmh and speed_kmh > max_speed_kmh:
        return None

    return speed_kmh


def open_csv_writer(csv_path: str):
    if not csv_path:
        return None, None

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "source",
            "track_id",
            "vehicle_class",
            "line_a_time_s",
            "line_b_time_s",
            "elapsed_s",
            "speed_kmh",
            "speed_limit_kmh",
            "overspeed",
        ],
    )
    writer.writeheader()
    return csv_file, writer


def write_speed_record(
    csv_writer,
    source,
    track_id: int,
    class_id: int,
    crossing_times,
    speed_kmh: float,
    speed_limit_kmh: float,
):
    if csv_writer is None:
        return

    line_a_time = crossing_times["A"]
    line_b_time = crossing_times["B"]
    csv_writer.writerow(
        {
            "source": str(source),
            "track_id": track_id,
            "vehicle_class": CLASS_NAMES_VI.get(class_id, class_id),
            "line_a_time_s": f"{line_a_time:.3f}",
            "line_b_time_s": f"{line_b_time:.3f}",
            "elapsed_s": f"{abs(line_b_time - line_a_time):.3f}",
            "speed_kmh": f"{speed_kmh:.1f}",
            "speed_limit_kmh": f"{speed_limit_kmh:.1f}" if speed_limit_kmh else "",
            "overspeed": bool(speed_limit_kmh and speed_kmh > speed_limit_kmh),
        }
    )


def make_output_path(source, output_path: str, output_dir: str):
    if output_path:
        return output_path

    if not output_dir or isinstance(source, int):
        return ""

    source_path = Path(source)
    return str(Path(output_dir) / f"{source_path.stem}_detected.mp4")


def run_single_source(
    model,
    source,
    confidence: float,
    tracker: str,
    output_path: str,
    show_window: bool,
    max_frames: int,
    line_a_spec,
    line_b_spec,
    detect_roi_only: bool,
    distance_meters: float,
    max_speed_kmh: float,
    speed_limit_kmh: float,
    csv_writer,
):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = open_video_writer(output_path, fps, (width, height))
    frame_count = 0
    track_history = defaultdict(lambda: deque(maxlen=30))
    last_centers = {}
    crossed_lines = defaultdict(dict)
    vehicle_speeds = {}
    vehicle_classes = {}
    speed_lines = (
        scale_line(line_a_spec, width, height),
        scale_line(line_b_spec, width, height),
    )
    roi_polygon = make_roi_polygon(speed_lines[0], speed_lines[1])

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detection_frame = apply_roi_mask(frame, roi_polygon) if detect_roi_only else frame
        results = model.track(
            detection_frame,
            persist=frame_count > 0,
            tracker=tracker,
            conf=confidence,
            classes=VEHICLE_CLASS_IDS,
            verbose=False,
        )[0]

        boxes = results.boxes
        if boxes is None:
            boxes = []

        draw_speed_lines(frame, speed_lines[0], speed_lines[1])
        time_seconds = frame_count / fps

        for detected_box in boxes:
            class_id = int(detected_box.cls[0])
            score = float(detected_box.conf[0])
            box = detected_box.xyxy[0].tolist()
            track_id = int(detected_box.id[0]) if detected_box.id is not None else None

            if track_id is not None:
                vehicle_classes[track_id] = class_id
                x1, y1, x2, y2 = map(int, box)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                previous_center = last_centers.get(track_id)
                last_centers[track_id] = center
                is_inside_roi = point_inside_polygon(center, roi_polygon)

                events = mark_line_crossings(
                    track_id=track_id,
                    previous_point=previous_center,
                    current_point=center,
                    speed_lines=speed_lines,
                    crossed_lines=crossed_lines,
                    time_seconds=time_seconds,
                )
                for event_track_id, line_name, event_time in events:
                    print(f"Track ID {event_track_id} crossed line {line_name} at {event_time:.2f}s")
                    speed_kmh = calculate_speed_kmh(
                        crossed_lines[event_track_id],
                        distance_meters,
                        max_speed_kmh,
                    )
                    if speed_kmh is not None and event_track_id not in vehicle_speeds:
                        vehicle_speeds[event_track_id] = speed_kmh
                        print(f"Track ID {event_track_id} speed: {speed_kmh:.1f} km/h")
                        write_speed_record(
                            csv_writer=csv_writer,
                            source=source,
                            track_id=event_track_id,
                            class_id=vehicle_classes.get(event_track_id, class_id),
                            crossing_times=crossed_lines[event_track_id],
                            speed_kmh=speed_kmh,
                            speed_limit_kmh=speed_limit_kmh,
                        )

                if not is_inside_roi:
                    continue

                speed_kmh = vehicle_speeds.get(track_id)
                is_overspeed = bool(speed_limit_kmh and speed_kmh and speed_kmh > speed_limit_kmh)
                draw_vehicle(frame, box, class_id, score, track_id, speed_kmh, is_overspeed)
                track_history[track_id].append(center)
                draw_track_trail(frame, track_history[track_id])
            else:
                x1, y1, x2, y2 = map(int, box)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                if point_inside_polygon(center, roi_polygon):
                    draw_vehicle(frame, box, class_id, score, track_id)

        if show_window:
            cv2.imshow("Vehicle Tracking - Press Q to quit", frame)

        if writer is not None:
            writer.write(frame)

        frame_count += 1
        if max_frames and frame_count >= max_frames:
            break

        if show_window:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()

    if show_window:
        cv2.destroyAllWindows()


def run_detection(
    source: str,
    model_path: str,
    confidence: float,
    tracker: str,
    output_path: str,
    output_dir: str,
    show_window: bool,
    max_frames: int,
    line_a_spec,
    line_b_spec,
    detect_roi_only: bool,
    distance_meters: float,
    max_speed_kmh: float,
    speed_limit_kmh: float,
    csv_path: str,
):
    model = YOLO(model_path)
    sources = get_video_files(source)
    csv_file, csv_writer = open_csv_writer(csv_path)

    try:
        if len(sources) > 1 and output_path:
            raise ValueError("Use --output-dir when --source is a folder with multiple videos.")

        for item in sources:
            item_output_path = make_output_path(item, output_path, output_dir)
            print(f"Processing: {item}")
            if item_output_path:
                print(f"Output: {item_output_path}")

            run_single_source(
                model=model,
                source=str(item) if not isinstance(item, int) else item,
                confidence=confidence,
                tracker=tracker,
                output_path=item_output_path,
                show_window=show_window,
                max_frames=max_frames,
                line_a_spec=line_a_spec,
                line_b_spec=line_b_spec,
                detect_roi_only=detect_roi_only,
                distance_meters=distance_meters,
                max_speed_kmh=max_speed_kmh,
                speed_limit_kmh=speed_limit_kmh,
                csv_writer=csv_writer,
            )
    finally:
        if csv_file is not None:
            csv_file.close()


def main():
    parser = argparse.ArgumentParser(description="Detect vehicles in a video or webcam stream.")
    parser.add_argument("--source", default="0", help="Video path or camera index. Example: 0 or videos/road.mp4")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model path. Default downloads yolov8n.pt.")
    parser.add_argument("--conf", type=float, default=None, help="Detection confidence threshold.")
    parser.add_argument("--tracker", default=None, help="Tracker config. Example: bytetrack.yaml or botsort.yaml")
    parser.add_argument("--output", default="", help="Optional output video path, e.g. outputs/detected.mp4")
    parser.add_argument("--output-dir", default="", help="Optional output folder for folder input.")
    parser.add_argument("--no-show", action="store_true", help="Process without opening a video window.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional limit for quick tests.")
    parser.add_argument(
        "--detect-roi-only",
        action="store_true",
        default=None,
        help="Mask everything outside the trapezoid before YOLO. Cleaner, but may miss small motorcycles.",
    )
    parser.add_argument(
        "--full-frame-detect",
        action="store_true",
        help="Force YOLO to inspect the full frame, overriding detect_roi_only in config.",
    )
    parser.add_argument(
        "--line-a",
        default=None,
        help="Line A as x1,y1,x2,y2 ratios. Default matches the sample trapezoid top line.",
    )
    parser.add_argument(
        "--line-b",
        default=None,
        help="Line B as x1,y1,x2,y2 ratios. Default matches the sample trapezoid bottom line.",
    )
    parser.add_argument(
        "--distance-meters",
        type=float,
        default=None,
        help="Real-world distance between line A and line B in meters.",
    )
    parser.add_argument(
        "--max-speed-kmh",
        type=float,
        default=None,
        help="Ignore calculated speeds above this value. Use 0 to disable filtering.",
    )
    parser.add_argument(
        "--speed-limit-kmh",
        type=float,
        default=None,
        help="Mark vehicles above this speed as overspeed. Use 0 to disable marking.",
    )
    parser.add_argument(
        "--csv",
        default="outputs/speed_results.csv",
        help="Path to save speed records as CSV. Use empty string to disable.",
    )
    parser.add_argument("--config", default="", help="Optional JSON config path.")
    parser.add_argument("--save-config", default="", help="Save the effective config to this JSON path and exit.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.line_a is not None:
        config["line_a"] = args.line_a
    if args.line_b is not None:
        config["line_b"] = args.line_b
    if args.distance_meters is not None:
        config["distance_meters"] = args.distance_meters
    if args.speed_limit_kmh is not None:
        config["speed_limit_kmh"] = args.speed_limit_kmh
    if args.max_speed_kmh is not None:
        config["max_speed_kmh"] = args.max_speed_kmh
    if args.conf is not None:
        config["confidence"] = args.conf
    if args.tracker is not None:
        config["tracker"] = args.tracker
    if args.detect_roi_only is not None:
        config["detect_roi_only"] = args.detect_roi_only
    if args.full_frame_detect:
        config["detect_roi_only"] = False

    if config["distance_meters"] <= 0:
        raise ValueError("--distance-meters must be greater than 0.")
    if config["max_speed_kmh"] < 0:
        raise ValueError("--max-speed-kmh must be 0 or greater.")
    if config["speed_limit_kmh"] < 0:
        raise ValueError("--speed-limit-kmh must be 0 or greater.")

    save_config(args.save_config, config)
    if args.save_config:
        print(f"Saved config: {args.save_config}")
        return

    run_detection(
        source=args.source,
        model_path=args.model,
        confidence=config["confidence"],
        tracker=config["tracker"],
        output_path=args.output,
        output_dir=args.output_dir,
        show_window=not args.no_show,
        max_frames=args.max_frames,
        line_a_spec=parse_line_spec(config["line_a"]),
        line_b_spec=parse_line_spec(config["line_b"]),
        detect_roi_only=config["detect_roi_only"],
        distance_meters=config["distance_meters"],
        max_speed_kmh=config["max_speed_kmh"],
        speed_limit_kmh=config["speed_limit_kmh"],
        csv_path=args.csv,
    )


if __name__ == "__main__":
    main()
