# Environment

- OS: Windows, Git Bash
- Python 3.11.9 (venv at .venv)
- MediaPipe 1.1.1: Tasks API only (`HolisticLandmarker`); legacy `mp.solutions` not available
- PyTorch 2.14.1+cpu locally; training will use Kaggle/Colab GPU

## Notes
- The Kaggle Isolated ASL Signs landmarks were extracted in 2023 with legacy Holistic.
  Our own extraction (ASL Citizen, live client) uses the Tasks HolisticLandmarker.
  Compare a few signs from both before mixing them (Phase 3).