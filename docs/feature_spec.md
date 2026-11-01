# Feature Spec v1 (frozen 2026-11-01)

Each frame becomes a vector of 320 values. A sequence becomes an array of shape (T, 320),
where T = number of frames.

## Layout (in this order)

| Block              | Points | Values | Contents |
|--------------------|--------|--------|----------|
| Pose               | 9      | 27     | nose 0, ears 7/8, shoulders 11/12, elbows 13/14, wrists 15/16 |
| Dominant hand      | 21     | 63     | hand landmarks 0-20 (0 = wrist, 4 per finger) |
| Non-dominant hand  | 21     | 63     | hand landmarks 0-20 |
| Face               | 43     | 129    | outer lips 20, inner lip centres 2, eyebrows 10, eyes 8, centres 3 |
| Relational         | -      | 36     | each wrist minus 6 face references (2 x 6 x 3) |
| Presence flags     | -      | 2      | dominant hand present, non-dominant hand present (1/0) |
| **Total**          |        | **320**|          |

Every point contributes x, y, z (3 values).

## Face points (43)
- Outer lips (20): 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375
- Inner lip centres (2): 13 (upper), 14 (lower)
- Eyebrows (10): 70, 63, 105, 66, 107 | 300, 293, 334, 296, 336
- Eyes (8): 33, 133, 159, 145 | 263, 362, 386, 374 (outer corner, inner corner, upper lid, lower lid)
- Centres (3): nose tip 1, forehead 10, chin 152

## Relational references (6)
nose 1, forehead 10, chin 152, right eye 133, left eye 362, mouth 0

## Mirror pairs (person's right <-> left)
- Pose: 7-8, 11-12, 13-14, 15-16 (0 unchanged)
- Hands: swap whole dominant/non-dominant blocks before assignment
- Face: 61-291, 185-409, 40-270, 39-269, 37-267, 146-375, 91-321, 181-405, 84-314,
  70-300, 63-293, 105-334, 66-296, 107-336, 33-263, 133-362, 159-386, 145-374
  (0, 17, 13, 14, 1, 10, 152 unchanged)

## Processing order
1. Load parquet -> (T, 543, 3), sorted by frame.
2. Select the landmarks above.
3. Dominant hand = hand visible (non-NaN) in more frames. Tie, including no hands -> right.
   If dominant is left: mirror (negate x, swap hand blocks, swap all pairs above).
4. Normalize: centre = shoulder midpoint, scale = distance nose -> centre,
   both averaged over the whole sequence.
5. Relational features from normalized coordinates.
6. NaN -> 0; presence flags record missing hands.