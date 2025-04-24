# utils/drawing_utils.py
import cv2

def draw_box_with_label(
    frame,
    box,
    label,
    color=(0,255,0),
    box_thickness=4,
    font_scale=1.5,
    font_thickness=3
):
    """
    Draws a colored bounding box with a filled label background.

    Args:
      frame:        np.ndarray, the image.
      box:          tuple(x1,y1,x2,y2)
      label:        str, text to display
      color:        BGR tuple for both box & label BG
      box_thickness:int, thickness of rectangle edges
      font_scale:   float, size of the font
      font_thickness:int, thickness of the text strokes
    """
    x1, y1, x2, y2 = box

    # 1) Draw the bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

    # 2) Compute text size so we can draw a BG rect
    (text_w, text_h), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        font_thickness
    )
    # make the background slightly larger than text
    cv2.rectangle(
        frame,
        (x1, y1 - text_h - baseline - 6),
        (x1 + text_w + 6, y1),
        color,
        thickness=-1
    )

    # 3) Overlay the text in white
    cv2.putText(
        frame,
        label,
        (x1 + 3, y1 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255,255,255),
        font_thickness,
        cv2.LINE_AA
    )
