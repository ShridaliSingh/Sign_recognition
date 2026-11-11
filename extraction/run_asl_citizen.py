import argparse
import os
import tempfile
import time
import zipfile
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

from extraction.extract import extract_video

VIDEO_PREFIX = "ASL_Citizen/videos/"
_zip = None
_model = None


def _init(zip_path, model_path):
    global _zip, _model
    _zip = zipfile.ZipFile(zip_path)
    _model = model_path


def _process(job):
    video_file, out_path = job
    if os.path.exists(out_path):
        return video_file, "skipped", 0.0
    start = time.time()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "video.mp4")
            with open(video_path, "wb") as f:
                f.write(_zip.read(VIDEO_PREFIX + video_file))
            arr = extract_video(video_path, _model)
        tmp_out = out_path + ".tmp.npy"
        np.save(tmp_out, arr.astype(np.float16))
        os.replace(tmp_out, out_path)
        return video_file, "ok", time.time() - start
    except Exception as e:
        return video_file, f"error: {e}", time.time() - start


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", default="data/asl_citizen/ASL_Citizen.zip")
    p.add_argument("--model", default="models/holistic_landmarker.task")
    p.add_argument("--out", default="data/asl_citizen/landmarks")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with zipfile.ZipFile(args.zip) as zf:
        tables = [pd.read_csv(zf.open(f"ASL_Citizen/splits/{s}.csv")) for s in ["train", "val", "test"]]
    files = pd.concat(tables)["Video file"].tolist()
    jobs = [(f, os.path.join(args.out, f.rsplit(".", 1)[0] + ".npy")) for f in files]
    if args.limit:
        jobs = jobs[:args.limit]

    print(f"{len(jobs)} videos, {args.workers} workers")
    errors, times, skipped = [], [], 0
    with Pool(args.workers, initializer=_init, initargs=(args.zip, args.model)) as pool:
        for video_file, status, secs in tqdm(pool.imap_unordered(_process, jobs, chunksize=4), total=len(jobs)):
            if status == "ok":
                times.append(secs)
            elif status == "skipped":
                skipped += 1
            else:
                errors.append((video_file, status))

    print(f"done: {len(times)} new, {skipped} skipped, {len(errors)} errors")
    if times:
        print(f"avg {np.mean(times):.2f} s per video per worker")
    if errors:
        pd.DataFrame(errors, columns=["video", "error"]).to_csv(
            os.path.join(args.out, "_errors.csv"), index=False)


if __name__ == "__main__":
    main()