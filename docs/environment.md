# Environment

- OS: Windows, Git Bash
- Python 3.11.9 (venv at .venv)
- MediaPipe 1.1.1: Tasks API only (`HolisticLandmarker`); legacy `mp.solutions` not available
- PyTorch 2.14.1+cpu locally; training will use Kaggle/Colab GPU

## Notes
- The Kaggle Isolated ASL Signs landmarks were extracted in 2023 with legacy Holistic.
  Our own extraction (ASL Citizen, live client) uses the Tasks HolisticLandmarker.
  Compare a few signs from both before mixing them (Phase 3).

# Kaggle Isolated ASL Signs: data notes

- train.csv: 94,477 sequences, 21 participants, 250 signs.
  Columns: path, participant_id, sequence_id, sign. One row = one recording.
- Sequences per participant: 3,338 to 4,968, fairly balanced (~18 per sign per signer).
- Parquet is long format: one row per (frame, landmark). Columns: frame, row_id, type,
  landmark_index, x, y, z. Types: face 468, pose 33, left_hand 21, right_hand 21 = 543 per frame.
  Every landmark has a row in every frame. Frame numbers are original video frames (don't start at 0).
- Missing hand = NaN in x, y, z for all 21 points at once (never partial). Face/pose present in every
  frame of the first sample. Sample 1 (participant 16069): right hand missing in 105/105 frames,
  left in 77/105 frames. Possibly mirrored labels; check across more files.
- x, y not limited to [0,1] (y up to ~2.6): MediaPipe estimates off-screen points (likely lower body).
  z ranges about -3.7 to 2.4: relative, noisy. Normalization handles scale; z is a testable flag.
- Hand labels differ by signer for the same sign ("cloud"):
  16069 -> left only (28/105 frames); 26734 -> right only (146/225); 2044 -> right only (8/16).  
  => Need dominant-hand handling / mirroring in Phase 1.
  Sequence length varies a lot (16 to 225 frames for the same sign).
- Frame-count distribution: to be computed on the full dataset in a Kaggle Notebook.