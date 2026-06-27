import time
from src.pipeline import run_pipeline

VIDEO = "data/test.mp4"
GROUND_TRUTH_IN = 49
GROUND_TRUTH_OUT = 32

def run_and_measure(label, **kwargs):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    start = time.time()
    cin, cout, unique = run_pipeline(video_path=VIDEO, display=False, save_stages=False, **kwargs)
    elapsed = time.time() - start
    return {"label": label, "in": cin, "out": cout, "unique": unique, "time": elapsed}

results = []
results.append(run_and_measure("Classic (MOG2 + CentroidTracker)", use_yolo=False))
results.append(run_and_measure("YOLO(s) + ByteTrack + LineCounter", use_yolo=True))

print(f"\n{'='*60}")
print(f"  BENCHMARK RESULTS")
print(f"{'='*60}")
print(f"{'Method':<42} {'IN':>4} {'OUT':>4} {'IDs':>5} {'Time':>8}")
print("-" * 60)
for r in results:
    print(f"{r['label']:<42} {r['in']:>4} {r['out']:>4} {r['unique']:>5} {r['time']:>7.1f}s")
print("-" * 60)
print(f"{'Ground truth (manual count)':<42} {'—':>4} {GROUND_TRUTH_OUT:>4}")