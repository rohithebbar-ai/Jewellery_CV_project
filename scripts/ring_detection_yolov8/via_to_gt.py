#!/usr/bin/env python3
import json
import argparse
import pickle
from collections import defaultdict
from pathlib import Path

def parse_via(via_path):
    via = json.load(open(via_path, 'r'))

    # Build fid → filename map
    fid2name = {
        int(v['fid']): v['fname']
        for v in via['file'].values()
    }

    # Output structures
    gt_boxes     = defaultdict(list)
    gt_assoc     = {}    # (fname, track_id) → finger_id
    gt_tracks    = defaultdict(list)
    tracks_by_id = defaultdict(list)

    for region in via['metadata'].values():
        # Get which image this belongs to
        vid = int(region.get('vid', -1))
        if vid not in fid2name:
            continue
        fname = fid2name[vid]

        # Extract the polygon coords, skip if none
        raw_xy = region.get('xy', [])
        pts_xy = raw_xy[1:]            # drop the leading “7”
        if len(pts_xy) < 2:
            # no (x,y) pairs → skip this region
            continue

        # Build (x,y) points
        pts = [(pts_xy[i], pts_xy[i+1]) for i in range(0, len(pts_xy), 2)]
        if not pts:
            continue

        # Compute bounding box
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

        # Parse your attributes (track_id=attr "1", finger_id=attr "2")
        av = region.get('av', {})
        try:
            track_id  = int(av.get('1', -1))
            finger_id = int(av.get('2', -1))
        except ValueError:
            track_id, finger_id = -1, -1

        # Store into your GT structures
        gt_boxes[fname].append([x1, y1, x2, y2])
        gt_assoc[(fname, track_id)] = finger_id
        gt_tracks[fname].append((track_id, [x1, y1, x2, y2]))
        tracks_by_id[track_id].append((fname, [x1, y1, x2, y2]))

    return gt_boxes, gt_assoc, gt_tracks, tracks_by_id

def main():
    p = argparse.ArgumentParser(
        description="Convert VIA JSON → GT for detection, association & tracking"
    )
    p.add_argument('--via', type=Path, required=True,
                   help="Path to your VIA project JSON")
    p.add_argument('--out', type=Path, default="processed_gt.pkl",
                   help="Where to pickle the GT dicts")
    args = p.parse_args()

    print(f"[INFO] Parsing VIA file {args.via} …")
    gt_boxes, gt_assoc, gt_tracks, tracks_by_id = parse_via(args.via)
    print(f"[INFO] Parsed {sum(len(v) for v in gt_boxes.values())} regions "
          f"across {len(gt_boxes)} frames, "
          f"{len(tracks_by_id)} unique track IDs.")

    print(f"[INFO] Saving to {args.out} …")
    with open(args.out, 'wb') as f:
        pickle.dump({
            'gt_boxes':     gt_boxes,
            'gt_assoc':     gt_assoc,
            'gt_tracks':    gt_tracks,
            'tracks_by_id': tracks_by_id
        }, f)

    print("[INFO] Done.")

if __name__ == '__main__':
    main()
