import numpy as np

class LineCounter:
    def __init__(self, line_y: int, max_missing: int = 30):
        self.line_y = line_y
        self.total_ids = 0
        self.max_missing = max_missing
        # {id: {"cx", "cy", "prev_cy", "bbox", "missing", "counted"}}
        self.tracks: dict[int, dict] = {}
        self.count_in = 0
        self.count_out = 0

    def update(self, detections: list[dict]) -> dict[int, dict]:
        seen = set()
        for d in detections:
            tid = d["id"]
            seen.add(tid)
            cx, cy = d["cx"], d["cy"]

            if tid not in self.tracks:
                self.total_ids += 1
                self.tracks[tid] = {
                    "cx": cx,
                    "cy": cy,
                    "prev_cy": cy,
                    "first_cy": cy,
                    "bbox": d["bbox"],
                    "missing": 0,
                    "counted": False,
                }
                continue

            t = self.tracks[tid]
            prev_cy = t["cy"]
            t.update({
                "cx": cx,
                "cy": cy,
                "prev_cy": prev_cy,
                "bbox": d["bbox"],
                "missing": 0,
            })

            if not t["counted"]:
                if prev_cy < self.line_y <= cy:
                    self.count_in += 1
                    t["counted"] = True
                elif prev_cy > self.line_y >= cy:
                    self.count_out += 1
                    t["counted"] = True

        for tid in list(self.tracks):
            if tid not in seen:
                self.tracks[tid]["missing"] += 1
                if self.tracks[tid]["missing"] > self.max_missing:
                    t = self.tracks[tid]
                    if not t["counted"]:
                        if t["first_cy"] < self.line_y <= t["cy"]:
                            self.count_in += 1
                        elif t["first_cy"] > self.line_y >= t["cy"]:
                            self.count_out += 1
                    del self.tracks[tid]

        return self.tracks

class CentroidTracker:
    """
    Simple centroid tracker: matches detections between frames by nearest distance.
    Tracks each person across frames and fires In/Out when they cross the counting line.
    """

    def __init__(self, line_y: int, max_distance: int = 80, max_missing: int = 5):
        self.line_y = line_y
        self.max_distance = max_distance
        self.max_missing = max_missing

        self.next_id = 0
        # {id: {"cx": int, "cy": int, "prev_cy": int, "missing": int, "counted": bool}}
        self.tracks: dict[int, dict] = {}

        self.count_in = 0   # crossed line top → bottom
        self.count_out = 0  # crossed line bottom → top

    def update(self, detections: list[dict]) -> dict[int, dict]:
        """
        Update tracks with new detections.
        Returns current tracks dict.
        """
        if not detections:
            # Mark all existing tracks as missing
            for tid in list(self.tracks):
                self.tracks[tid]["missing"] += 1
                if self.tracks[tid]["missing"] > self.max_missing:
                    del self.tracks[tid]
            return self.tracks

        det_centers = np.array([[d["cx"], d["cy"]] for d in detections])

        if not self.tracks:
            for d in detections:
                self._register(d["cx"], d["cy"])
            return self.tracks

        track_ids = list(self.tracks)
        track_centers = np.array([[self.tracks[tid]["cx"], self.tracks[tid]["cy"]] for tid in track_ids])

        # Compute pairwise distances
        distances = np.linalg.norm(
            track_centers[:, np.newaxis, :] - det_centers[np.newaxis, :, :], axis=2
        )  # shape: (n_tracks, n_dets)

        matched_tracks = set()
        matched_dets = set()

        # Greedy matching: closest pairs first
        while True:
            if distances.size == 0:
                break
            idx = np.unravel_index(np.argmin(distances), distances.shape)
            t_idx, d_idx = idx
            if distances[t_idx, d_idx] > self.max_distance:
                break
            tid = track_ids[t_idx]
            if tid in matched_tracks or d_idx in matched_dets:
                distances[t_idx, d_idx] = np.inf
                continue

            prev_cy = self.tracks[tid]["cy"]
            new_cy = detections[d_idx]["cy"]

            self.tracks[tid].update({
                "cx": detections[d_idx]["cx"],
                "prev_cy": prev_cy,
                "cy": new_cy,
                "missing": 0,
            })

            # Check line crossing
            if not self.tracks[tid]["counted"]:
                if prev_cy < self.line_y <= new_cy:
                    self.count_in += 1
                    self.tracks[tid]["counted"] = True
                elif prev_cy > self.line_y >= new_cy:
                    self.count_out += 1
                    self.tracks[tid]["counted"] = True

            matched_tracks.add(tid)
            matched_dets.add(d_idx)
            distances[t_idx, :] = np.inf
            distances[:, d_idx] = np.inf

        # Remove stale tracks
        for i, tid in enumerate(track_ids):
            if tid not in matched_tracks:
                self.tracks[tid]["missing"] += 1
                if self.tracks[tid]["missing"] > self.max_missing:
                    del self.tracks[tid]

        # Register new detections that weren't matched
        for d_idx, d in enumerate(detections):
            if d_idx not in matched_dets:
                self._register(d["cx"], d["cy"])

        return self.tracks

    def _register(self, cx: int, cy: int):
        self.tracks[self.next_id] = {
            "cx": cx, "cy": cy, "prev_cy": cy,
            "missing": 0, "counted": False,
        }
        self.next_id += 1
