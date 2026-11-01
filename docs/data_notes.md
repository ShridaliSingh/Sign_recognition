# Kaggle Isolated ASL Signs: data notes
- train.csv: 94,477 sequences, 21 participants, 250 signs.
  Columns: path, participant_id, sequence_id, sign. One row = one recording.
- Sequences per participant: 3,338 to 4,968, fairly balanced (~18 per sign per signer).
- Parquet is long format: one row per (frame, landmark). Columns: frame, row_id, type,
  landmark_index, x, y, z. Types: face 468, pose 33, left_hand 21, right_hand 21 = 543 per frame.
  Every landmark has a row in every frame. Frame numbers are original video frames (don't start at 0).
- Missing hand = NaN in x, y, z for all 21 points at once (never partial). Face/pose present in every
  frame of the first sample. Sample 1 (participant 16069): right hand missing in 105/105 frames,
  left in 77/105 frames. See hand-label finding below.
- x, y not limited to [0,1] (y up to ~2.6): MediaPipe estimates off-screen points (likely lower body).
  z ranges about -3.7 to 2.4: relative, noisy. Normalization handles scale; z is a testable flag.
- Hand labels differ by signer for the same sign ("cloud"):
  16069 -> left only (28/105 frames); 26734 -> right only (146/225); 2044 -> right only (8/16).  
  => Need dominant-hand handling / mirroring in Phase 1.
  Sequence length varies a lot (16 to 225 frames for the same sign).
- Frame counts (all 94,477): min 2, median 22, mean 38, 90th pct 92, 95th pct 135,
  99th pct 219, max 537. 17,640 (~19%) under 10 frames; 1,526 (~1.6%) over 200.
  => Phase 2: max length ~96-128; resample longer sequences rather than truncate.
  Short sequences are common and must be handled, not dropped.

## ASLLRP (DAI 2): terms
- Site: http://dai.cs.rutgers.edu/dai/s/dai
- Research and education use only. No redistribution without permission.
  No commercial use without explicit permission.
- Publications/presentations must cite them and include URLs:
  http://dai.cs.rutgers.edu/, http://dai.cs.rutgers.edu/dai/s/signbank, http://www.bu.edu/asllrp/
- Citations to use (from their Terms of Use page):
  - Neidle & Opoku (2020), User's Guide to the ASLLRP DAI 2, Version 2. ASLLRP Report No. 18, BU.
  - Neidle, Opoku & Metaxas (2022), ASL Video Corpora & Sign Bank. arXiv:2201.07899.
  - Neidle, Opoku, Dimitriadis & Metaxas (2018), NEW Shared & Interconnected ASL Resources. LREC 2018 workshop, pp. 147-154.
- DawnSignPress data has been withdrawn (no longer viewable or downloadable).