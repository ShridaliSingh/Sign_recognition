# Where each landmark type starts inside the 543 rows of one frame (Kaggle order)
FACE_START = 0
LEFT_HAND_START = 468
POSE_START = 489
RIGHT_HAND_START = 522

# Points we keep, by MediaPipe's own numbering within each type
POSE = [0, 7, 8, 11, 12, 13, 14, 15, 16]
HAND = list(range(21))
LIPS_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
              291, 146, 91, 181, 84, 17, 314, 405, 321, 375]
LIPS_INNER = [13, 14]
EYEBROWS = [70, 63, 105, 66, 107, 300, 293, 334, 296, 336]
EYES = [33, 133, 159, 145, 263, 362, 386, 374]
MIDLINE = [1, 10, 152]
FACE = LIPS_OUTER + LIPS_INNER + EYEBROWS + EYES + MIDLINE

# Positions in the 543 rows, in our order: pose, right hand, left hand, face
SELECTED = (
    [POSE_START + i for i in POSE]
    + [RIGHT_HAND_START + i for i in HAND]
    + [LEFT_HAND_START + i for i in HAND]
    + [FACE_START + i for i in FACE]
)

# Where each block sits in the selected (94-point) array
POSE_BLOCK = slice(0, len(POSE))
DOM_HAND_BLOCK = slice(POSE_BLOCK.stop, POSE_BLOCK.stop + 21)
NONDOM_HAND_BLOCK = slice(DOM_HAND_BLOCK.stop, DOM_HAND_BLOCK.stop + 21)
FACE_BLOCK = slice(NONDOM_HAND_BLOCK.stop, NONDOM_HAND_BLOCK.stop + len(FACE))
N_POINTS = FACE_BLOCK.stop

# Left/right pairs (MediaPipe numbering) that swap when mirroring
POSE_PAIRS = [(7, 8), (11, 12), (13, 14), (15, 16)]
FACE_PAIRS = [(61, 291), (185, 409), (40, 270), (39, 269), (37, 267),
              (146, 375), (91, 321), (181, 405), (84, 314),
              (70, 300), (63, 293), (105, 334), (66, 296), (107, 336),
              (33, 263), (133, 362), (159, 386), (145, 374)]


def _build_flip_order():
    order = list(range(N_POINTS))
    for k in range(21):
        a = DOM_HAND_BLOCK.start + k
        b = NONDOM_HAND_BLOCK.start + k
        order[a], order[b] = b, a
    for p, q in POSE_PAIRS:
        a = POSE_BLOCK.start + POSE.index(p)
        b = POSE_BLOCK.start + POSE.index(q)
        order[a], order[b] = b, a
    for p, q in FACE_PAIRS:
        a = FACE_BLOCK.start + FACE.index(p)
        b = FACE_BLOCK.start + FACE.index(q)
        order[a], order[b] = b, a
    return order


FLIP_ORDER = _build_flip_order()

# Positions in the selected array of the points normalization needs
NOSE = POSE_BLOCK.start + POSE.index(0)
L_SHOULDER = POSE_BLOCK.start + POSE.index(11)
R_SHOULDER = POSE_BLOCK.start + POSE.index(12)

# Relational references: nose, forehead, chin, right eye, left eye, mouth
REL_REFS = [FACE_BLOCK.start + FACE.index(i) for i in [1, 10, 152, 133, 362, 0]]

