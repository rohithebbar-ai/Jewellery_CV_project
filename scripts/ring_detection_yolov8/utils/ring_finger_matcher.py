"""# utils/ring_finger_matcher.py
import math

def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def find_closest_landmark(ring_center, hand_landmarks):
    
    ring_center: (x_norm, y_norm) in [0–1] coords
    hand_landmarks: list of mp.solutions.hands.HandLandmark lists
    Returns: (hand_index, landmark_index) or (None, None)
    
    best = (None, None)
    min_dist = float('inf')
    for h_idx, hand in enumerate(hand_landmarks):
        for lm_idx, lm in enumerate(hand.landmark):
            dist = euclidean(ring_center, (lm.x, lm.y))
            if dist < min_dist:
                min_dist, best = dist, (h_idx, lm_idx)
    return best
"""

# utils/ring_finger_matcher.py
import math

def euclidean_3d(p1, p2):
    """3D Euclidean distance."""
    return math.sqrt(
        (p1[0] - p2[0])**2 +
        (p1[1] - p2[1])**2 +
        (p1[2] - p2[2])**2
    )

def finger_direction(hand, lm_idx):
    """
    Compute a unit vector for the finger segment at lm_idx.
    We'll use the vector from the PIP to TIP for each finger,
    e.g. for lm_idx 8 (index fingertip), segment = lm[8] - lm[6].
    """
    # map TIP→PIP indices according to MediaPipe’s schema
    PIP_IDX = {4: 3, 8: 6, 12: 10, 16: 14, 20: 18}
    if lm_idx not in PIP_IDX:
        return None
    tip = hand.landmark[lm_idx]
    pip = hand.landmark[PIP_IDX[lm_idx]]
    vx, vy, vz = tip.x - pip.x, tip.y - pip.y, tip.z - pip.z
    norm = math.sqrt(vx*vx + vy*vy + vz*vz)
    return (vx/norm, vy/norm, vz/norm)

def find_closest_landmark_3d(ring_center, hand_landmarks, angle_thresh=0.6):
    """
    ring_center: (x_norm, y_norm)  — 2D center of box in [0..1]
    hand_landmarks: list of mp.HandLandmarks
    Returns: (hand_idx, lm_idx) or (None, None)
    """
    best = (None, None)
    best_score = float('inf')
    for h_idx, hand in enumerate(hand_landmarks):
        for lm_idx, lm in enumerate(hand.landmark):
            # 1) 3D distance: use lm.z as depth
            dist3d = euclidean_3d(
                (ring_center[0], ring_center[1], 0.0),  # assume ring z~0
                (lm.x, lm.y, lm.z)
            )

            # 2) orientation check: is ring “in front of” this finger?
            finger_vec = finger_direction(hand, lm_idx)
            if finger_vec is not None:
                # ring direction from joint → ring center
                rx, ry, rz = ring_center[0] - lm.x, ring_center[1] - lm.y, -lm.z
                rnorm = math.sqrt(rx*rx + ry*ry + rz*rz)
                ring_vec = (rx/rnorm, ry/rnorm, rz/rnorm)
                # cosine similarity
                cosang = (finger_vec[0]*ring_vec[0] +
                          finger_vec[1]*ring_vec[1] +
                          finger_vec[2]*ring_vec[2])
                # if it’s pointing roughly along the finger’s axis, boost it
                if cosang < angle_thresh:
                    # penalize unlikely matches
                    dist3d *= 2.0

            # pick the smallest “score”
            if dist3d < best_score:
                best_score, best = dist3d, (h_idx, lm_idx)

    return best
