"""Bounded CPU decoding for the prepared EE58 MP4 dataset.

This preserves UniT's torchvision/PyAV timestamp rule: select the closest
decoded frame at or before each requested timestamp, falling back to the first
decoded frame. It consumes packed-video timestamps supplied by EE58Dataset;
it does not calculate offsets, alter RGB channels, or touch image-bit decoding.
"""
from __future__ import annotations

import av
import numpy as np


def get_frames_by_timestamps_bounded(video_path: str, timestamps) -> np.ndarray:
    """Return uint8 [T,H,W,3] RGB with one codec thread and O(T) RGB storage.

    EE58's current/goal queries are finite and nondecreasing, including clipped
    duplicate timestamps. Reject malformed queries instead of returning a
    silently different set of frames. Decoder and codec resources are closed
    on success, EOF, and exceptions; native codec errors retain their traceback.
    """
    targets = np.asarray(timestamps, dtype=np.float64)
    if targets.ndim != 1 or not len(targets) or not np.isfinite(targets).all():
        raise ValueError("Video timestamps must be a nonempty finite 1D sequence")
    if np.any(np.diff(targets) < 0):
        raise ValueError("EE58 current/goal timestamps must be nondecreasing")

    container = None
    try:
        container = av.open(video_path, metadata_errors="ignore")
        stream = container.streams.video[0]
        # torchvision's pyav backend ignores VideoReader(num_threads=...).
        # Set the codec itself before its first decode/open, avoiding AUTO's
        # per-reader CPU thread pools across 32 persistent DataLoader workers.
        stream.codec_context.thread_count = 1
        stream.codec_context.thread_type = "SLICE"
        offset = int(round(max(0.0, float(targets[0])) / stream.time_base))
        container.seek(offset, backward=True, any_frame=False, stream=stream)

        selected = []
        previous = None
        previous_ts = None
        cached_frame = None
        cached_rgb = None

        def retain(frame):
            nonlocal cached_frame, cached_rgb
            if frame is not cached_frame:
                cached_rgb = frame.to_rgb().to_ndarray()
                cached_frame = frame
            selected.append(cached_rgb)

        for frame in container.decode(video=0):
            if frame.pts is None or frame.time_base is None:
                raise ValueError(f"Decoded video frame has no timestamp: {video_path}")
            current_ts = float(frame.pts * frame.time_base)
            while len(selected) < len(targets) and targets[len(selected)] < current_ts:
                retain(previous if previous is not None else frame)
            while len(selected) < len(targets) and targets[len(selected)] == current_ts:
                retain(frame)
            if len(selected) == len(targets):
                break
            # Native argmin picks the first frame if equal PTS values occur.
            if previous_ts is None or current_ts > previous_ts:
                previous, previous_ts = frame, current_ts

        if previous is None and not selected:
            raise ValueError(f"No video frames decoded: {video_path}")
        while len(selected) < len(targets):
            retain(previous)
        return np.stack(selected, axis=0)
    finally:
        if container is not None:
            container.close()
