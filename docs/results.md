# Results

All runs: Kaggle Isolated ASL Signs, 250 signs, features v1 (320/frame),
signer-independent split from training/splits.json (14 train / 3 val / 4 test signers).
Accuracy = top-1 on the validation signers unless stated otherwise.

## Exp 1: Pipeline check, 1D conv baseline 
- Model: projection 320->256, 2x Conv1d (kernel 5, receptive field 9 frames), masked mean pooling,
  dropout 0.3, linear to 250.
- Training: Adam lr 1e-3, batch 64, 15 epochs, max length 128, no augmentation. T4 GPU, 19 s/epoch.
- Result: best val acc 41.8% (epoch 14). Train acc 61% at epoch 15.
  Val loss lowest at epoch 9 (3.30), then flat/noisy while train loss kept falling.
- Notes: Pipeline end-to-end verified (chance = 0.4%). Large train/val gap suggests overfitting
  to training signers. Next: augmentation, then transformer.

## Exp 2: Conv baseline + augmentation
- Same as Exp 1, but training set augmented: speed x0.8-1.2, scale x0.9-1.1, rotation ±10° (x-y plane).
  Val/test never augmented. 21 s/epoch.
- Result: best val acc 41.5% (epoch 13), vs 41.8% in Exp 1, no measurable difference.
  Train acc 57% at epoch 15 (vs 61%); gap shrank ~22 -> ~18 points, but via lower train acc, not higher val acc.
  Lowest val loss 3.31 (epoch 10). Val loss zig-zags between epochs -> possibly lr too high.
- Interpretation: model capacity is the bottleneck (only 61% train acc without aug), so a regularizer
  like augmentation has little to fix. Train acc still rising at epoch 15. Keep augmentation on;
  revisit with a stronger model and longer training.
- Next: transformer encoder, with lr schedule.