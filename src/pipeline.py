import cv2
import numpy as np
from pathlib import Path

from src.enhance import enhance_clahe
from src.segment import create_subtractor, segment_mog2
from src.clean import clean_mask
from src.detect import detect_people
from src.detect_yolo import detect_people_yolo
from src.decision import CentroidTracker
from src.visualize import visualize


def save_stage(output_dir: Path, frame_idx: int, name: str, image: np.ndarray):
    path = output_dir / f"{frame_idx:05d}_{name}.jpg"
    cv2.imwrite(str(path), image)


def run_pipeline(
    video_path: str,
    output_dir: str = "data/output",
    line_y: int | None = None,
    min_area: int = 800,
    max_area: int = 18000,
    kernel_size: int = 7,
    warmup_frames: int = 120,
    save_stage_interval: int = 300,
    save_stages: bool = True,
    display: bool = False,
    use_yolo: bool = False,
    yolo_weights: str = "yolov8n.pt",
    yolo_conf: float = 0.35,
    motion_threshold: int = 5000,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if line_y is None:
        line_y = frame_h // 2

    subtractor = create_subtractor()
    tracker = CentroidTracker(line_y=line_y)

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(out_path / "result.mp4"), fourcc, fps, (frame_w, frame_h)
    )

    frame_idx = 0
    print(f"Processing: {video_path}  |  line_y={line_y}  |  size={frame_w}x{frame_h}")

    # Warm up MOG2 background model before tracking starts
    print(f"Warming up MOG2 for {warmup_frames} frames...")
    for _ in range(warmup_frames):
        ret, frame = cap.read()
        if not ret:
            break
        subtractor.apply(enhance_clahe(frame))
    print("Warmup done. Starting tracking...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Stage 1: Enhance
        enhanced = enhance_clahe(frame)

        # Stage 2: Segment
        raw_mask = segment_mog2(enhanced, subtractor)

        # Stage 3: Clean
        clean = clean_mask(raw_mask, kernel_size=kernel_size)

        # Stage 4: Detect
        if use_yolo:
            yolo_result = detect_people_yolo(
                frame, clean,
                motion_threshold=motion_threshold,
                conf=yolo_conf,
                weights=yolo_weights,
            )
            if yolo_result is None:
                # No motion — skip tracker update, keep last result
                writer.write(result if frame_idx > 0 else frame)
                frame_idx += 1
                continue
            detections = yolo_result
        else:
            detections = detect_people(clean, min_area=min_area, max_area=max_area)

        # Stage 5: Decision / tracking
        tracks = tracker.update(detections)

        # Stage 6: Visualize
        result = visualize(frame, detections, tracks, tracker.count_in, tracker.count_out, line_y)

        writer.write(result)

        if save_stages and frame_idx % save_stage_interval == 0:
            save_stage(out_path, frame_idx, "1_original", frame)
            save_stage(out_path, frame_idx, "2_enhanced", enhanced)
            save_stage(out_path, frame_idx, "3_raw_mask", raw_mask)
            save_stage(out_path, frame_idx, "4_clean_mask", clean)
            # Detection frame
            det_frame = frame.copy()
            for d in detections:
                x, y, w, h = d["bbox"]
                cv2.rectangle(det_frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            save_stage(out_path, frame_idx, "5_detection", det_frame)
            save_stage(out_path, frame_idx, "6_final", result)

        if display:
            cv2.imshow("People Counter", result)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    writer.release()
    if display:
        cv2.destroyAllWindows()

    print(f"Done. Frames processed: {frame_idx}")
    print(f"Count IN:  {tracker.count_in}")
    print(f"Count OUT: {tracker.count_out}")
    print(f"Output saved to: {out_path}")
    return tracker.count_in, tracker.count_out
