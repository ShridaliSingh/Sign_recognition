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

## Exp 3: Transformer encoder
- Model: proj 320->192 + LayerNorm, learned positional embedding, 4x TransformerEncoderLayer
  (d_model 192, 4 heads, ff 384, dropout 0.2, pre-norm), masked mean pooling, linear to 250.
  1,323,322 parameters.
- Training: AdamW lr 1e-3, weight decay 0.05, OneCycleLR (10% warmup), grad clip 1.0,
  40 epochs, batch 64, augmentation as Exp 2. T4 GPU, 53 s/epoch (~35 min).
- Result: best val acc 58.4% (epoch 35), +16.6 points over Exp 1. Train acc 94.7% at epoch 40.
  Lowest val loss 2.22 (epochs 11, 24); afterwards val loss rose to 2.62 while val acc held ~58%
  (overconfidence).
- Interpretation: global attention fixed the conv bottleneck; model now overfits heavily
  (~37-point gap). Next levers: regularization and stronger augmentation.
- Caveat: model, optimizer, schedule and epochs all changed vs Exp 2. Conv with this recipe
  would isolate the architecture's effect.

  ### Exp 3 error analysis (validation signers)
- Top-1 58.35% (matches training log), top-5 80.94%: ~23% of cases are near misses.
- ~54 recordings per sign per split; e.g. dryer->dry 39 means ~72% of "dryer" called "dry".
- Most confused: dryer->dry 39, lips->mouth 26, awake->wake 26, duck->goose 22, kitty->cat 22,
  stay->that 21, listen->hear 21, man->dad 19, pen->pencil 17, book->open 17, goose->duck 16,
  now->if 15, wake->awake 15, alligator->rain 15, brother->boy 14.
- Groups: (1) near-identical signs/synonyms (awake/wake, kitty/cat, listen/hear) - label ceiling,
  errors go both ways; (2) same location, small difference (man/dad, brother/boy, lips/mouth);
  (3) same handshape, fine finger detail (duck/goose, pen/pencil, stay/that) - landmark weak spot.
- Idea: report accuracy with synonym groups merged as a fairer meaning-level metric.

## Exp 4: Transformer + stronger regularization
- Same as Exp 3, plus: frame dropout 0.1, hand dropout 0.1 (per frame, visible hands only),
  model dropout 0.3. 51 s/epoch.
- Result: best val acc 58.57% (epoch 37), vs 58.35%, no measurable gain.
  Train acc 87.6% (vs 94.7%); gap ~37 -> ~29 points. Final val loss 2.43 (vs 2.62): less overconfident.
- Interpretation: Exp 2 and 4 both cut train acc without raising val acc -> ceiling (~58%) is not
  from overfitting. Likely causes: only 14 training signers (limited style variety) and label
  ceiling (near-identical signs, fine finger detail). Next lever: more signers (ASL Citizen, 52).
- Chosen as final Phase 2 model (best val acc, better calibrated).

- Local checkpoints (models/, not in git): Exp 3 -> transformer_best.pt, Exp 4 -> transformer_reg_best.pt (final Phase 2 model).

## Final test result: Transformer with regularization (Exp 4) (2026-11-07)
Model: Exp 4 (transformer_reg_best.pt), chosen by best validation accuracy.
Test set used once, no tuning afterwards.
- Val: top-1 58.57%, top-5 81.80%
- **Test: top-1 60.73%, top-5 83.32%** (4 unseen signers, 17,651 sequences, 250 signs)
- Per signer (top-1): test 30680 40.0%, 37779 70.0%, 49445 55.4%, 62590 71.9%;
  val 27610 59.0%, 34503 49.3%, 37055 67.2%. Spread across people ~32 points.
- Most confused (test): awake->wake 36, hear->ear 31, jump->stairs 31, wake->awake 31,
  cat->kitty 28, wolf->sleepy 27, grass->taste 26, mouth->lips 23, puppy->dog 22, home->kiss 22,
  wait->finger 22, dryer->dry 21, where->up 21, goose->bird 21, chin->thirsty 20.
- Conclusion: same error categories as val (synonyms, same face location, fine handshape).
  Per-signer spread dwarfs regularization effects -> signer variety is the main limit.