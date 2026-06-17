import time
import statistics
from pathlib import Path

import torch
import torchvision
from torchcodec.decoders import VideoDecoder

videos = sorted(
    Path("datasets/SARM_manual_test_10Episodes_lerobotv3.0")
    .glob("videos/*/chunk-000/*.mp4")
)[:40]

print("videos", len(videos))
print("first", videos[0] if videos else None)


def decode_tv_pyav(video_path, timestamps, tolerance_s=0.1):
    torchvision.set_video_backend("pyav")
    reader = torchvision.io.VideoReader(str(video_path), "video")

    first_ts = min(timestamps)
    last_ts = max(timestamps)
    reader.seek(first_ts, keyframes_only=True)

    loaded_frames = []
    loaded_ts = []
    try:
        for frame in reader:
            current_ts = float(frame["pts"])
            loaded_frames.append(frame["data"])
            loaded_ts.append(current_ts)
            if current_ts >= last_ts:
                break
    finally:
        reader.container.close()

    query_ts = torch.tensor(timestamps, dtype=torch.float32)
    loaded_ts_tensor = torch.tensor(loaded_ts, dtype=torch.float32)
    dist = torch.cdist(query_ts[:, None], loaded_ts_tensor[:, None], p=1)
    min_dist, argmin = dist.min(1)

    if not (min_dist <= tolerance_s).all():
        raise RuntimeError(f"tolerance miss {min_dist.max().item()} for {video_path}")

    return torch.stack([loaded_frames[int(idx)] for idx in argmin])


def decode_torchcodec_fresh(video_path, timestamps):
    decoder = VideoDecoder(str(video_path))
    if len(timestamps) == 1:
        return decoder.get_frame_played_at(float(timestamps[0])).data.unsqueeze(0)
    # breakpoint()
    return decoder.get_frames_played_at([float(t) for t in timestamps]).data


def bench(name, decode_fn, timestamps, rounds=3):
    for path in videos[:3]:
        decode_fn(path, timestamps)

    durations = []
    for _ in range(rounds):
        start = time.perf_counter()
        for path in videos:
            out = decode_fn(path, timestamps)
            _ = out.shape
        durations.append(time.perf_counter() - start)

    mean = statistics.mean(durations)
    print(f"{name}: rounds={rounds} videos/round={len(videos)} frames/round={len(videos) * len(timestamps)}")
    print(f"  times_s={[round(x, 4) for x in durations]}")
    print(
        f"  mean_s={mean:.4f} "
        f"videos_per_s={len(videos) / mean:.2f} "
        f"frames_per_s={(len(videos) * len(timestamps)) / mean:.2f} "
        f"ms_per_video={mean / len(videos) * 1000:.2f}"
    )


single = [0.0]
window = [i / 30 for i in range(10)]

bench("pyav_single_t0", decode_tv_pyav, single)
bench("pyav_10frames_0_to_0.3s", decode_tv_pyav, window)
bench("torchcodec_fresh_single_t0", decode_torchcodec_fresh, single)
bench("torchcodec_fresh_10frames_0_to_0.3s", decode_torchcodec_fresh, window)