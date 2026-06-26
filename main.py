import argparse
from src.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="People Counting & Flow Pipeline")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--output", default="data/output", help="Directory to save output files")
    parser.add_argument("--line-y", type=int, default=None, help="Y coordinate of counting line (default: frame center)")
    parser.add_argument("--min-area", type=int, default=500, help="Minimum contour area to count as a person")
    parser.add_argument("--max-area", type=int, default=50000, help="Maximum contour area (filters out vehicles)")
    parser.add_argument("--kernel-size", type=int, default=5, help="Morphological kernel size for mask cleaning")
    parser.add_argument("--display", action="store_true", help="Show live preview window")
    parser.add_argument("--no-save-stages", action="store_true", help="Skip saving per-stage images")
    parser.add_argument("--yolo", action="store_true", help="Use YOLO for detection (MOG2 as motion gate)")
    parser.add_argument("--yolo-weights", default="yolov8n.pt", help="YOLO weights file")
    parser.add_argument("--yolo-conf", type=float, default=0.35, help="YOLO confidence threshold")
    parser.add_argument("--motion-threshold", type=int, default=5000, help="Min motion area (px) to trigger YOLO")
    args = parser.parse_args()

    run_pipeline(
        video_path=args.video,
        output_dir=args.output,
        line_y=args.line_y,
        min_area=args.min_area,
        max_area=args.max_area,
        kernel_size=args.kernel_size,
        save_stages=not args.no_save_stages,
        display=args.display,
        use_yolo=args.yolo,
        yolo_weights=args.yolo_weights,
        yolo_conf=args.yolo_conf,
        motion_threshold=args.motion_threshold,
    )


if __name__ == "__main__":
    main()