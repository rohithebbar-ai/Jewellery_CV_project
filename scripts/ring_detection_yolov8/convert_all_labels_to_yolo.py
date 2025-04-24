import os, glob

# Folders to scan 
LABELS_DIRS = [
    'data/train/labels',
    'data/valid/labels'
]

for lbl_dir in LABELS_DIRS:
    for path in glob.glob(os.path.join(lbl_dir, '*.txt')):
        new_lines = []
        for L in open(path):
            parts = L.strip().split()
            if not parts:
                continue
            
            # Case A : 8-point polygon format (1 + 8 numbers)
            if len(parts) == 1 + 8:
                # ignore original class (parts[0]), remap to 0
                coords = list(map(float, parts[1:]))
                xs = coords[0::2]
                ys = coords[1::2]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                x_c = (x_min + x_max) / 2
                y_c = (y_min + y_max) / 2
                w   = x_max - x_min
                h   = y_max - y_min
                new_lines.append(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
                
            # Case B : already YOLO-style but wrong class (1 + 4 numbers)
            elif len(parts) == 1 + 4:
                # remap any class != 0 to 0
                _, *coords = parts
                new_lines.append("0 " + " ".join(coords))

            # anything else we skip
            else:
                continue

        # overwrite file
        with open(path, 'w') as f:
            if new_lines:
                f.write("\n".join(new_lines) + "\n")
        print(f"[UPDATED] {path} → {len(new_lines)} box(es)")

print("All labels converted to class 0, YOLO 5‑number format.")
