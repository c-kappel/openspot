import cv2 as cv
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import threading

SPOT_2 = (40, 400, 2070, 640)

SPOTS = [SPOT_2]

# Anchor points mapping camera pixels to map.png pixels
CAM_ANCHOR_1 = (10,  230)
CAM_ANCHOR_2 = (570, 240)
MAP_ANCHOR_1 = (1000, 1300)
MAP_ANCHOR_2 = (1010, 1320)


def cam_to_map(cam_x, cam_y):
    tx = (cam_x - CAM_ANCHOR_1[0]) / (CAM_ANCHOR_2[0] - CAM_ANCHOR_1[0])
    ty = (cam_y - CAM_ANCHOR_1[1]) / (CAM_ANCHOR_2[1] - CAM_ANCHOR_1[1])
    mx = MAP_ANCHOR_1[0] + tx * (MAP_ANCHOR_2[0] - MAP_ANCHOR_1[0])
    my = MAP_ANCHOR_1[1] + ty * (MAP_ANCHOR_2[1] - MAP_ANCHOR_1[1])
    return (mx, my)


def free_segments_to_map(spots, free_segments):
    """
    Converts free camera pixel segments to map.png pixel rectangles.

    Returns:
        dict: spot_index -> list of {"topLeft": [mx, my], "bottomRight": [mx, my]}
    """
    result = {}
    for spot_i, segs in free_segments.items():
        sx1, sy1, sx2, sy2 = spots[spot_i]
        result[spot_i] = [
            {"topLeft": cam_to_map(x1, sy1), "bottomRight": cam_to_map(x2, sy2)}
            for (x1, x2) in segs
        ]
    return result

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
)
latest_free_map = {}

@app.get("/parking")
def get_parking():
    return latest_free_map

threading.Thread(
    target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000),
    daemon=True,
).start()

cap = cv.VideoCapture("/Users/christiankappel/Projects/openspot/footage/test3_1080p.mov")
fps = cap.get(cv.CAP_PROP_FPS) or 30
frame_delay = int(1000 / fps)

model = YOLO("yolov8n.pt")

position_history = defaultdict(list)
moving_streak    = defaultdict(int)

HISTORY_FRAMES     = 10
MOVEMENT_THRESHOLD = 5
SUSTAIN_FRAMES     = 5
MIN_CONF           = 0.25


def get_free_segments(spots, parked_boxes):
    """
    For each spot, find parked cars whose center falls within it and return
    the x-segments of the spot that are still free to park in.

    Returns:
        dict: spot_index -> list of (x1, x2) free pixel ranges
    """
    result = {}
    for i, (sx1, sy1, sx2, sy2) in enumerate(spots):
        free = [(sx1, sx2)]
        for (bx1, by1, bx2, by2) in parked_boxes:
            cx = (bx1 + bx2) / 2
            cy = (by1 + by2) / 2
            if sx1 <= cx <= sx2 and sy1 <= cy <= sy2:
                ox1 = int(max(bx1, sx1))
                ox2 = int(min(bx2, sx2))
                new_free = []
                for (fx1, fx2) in free:
                    if ox2 <= fx1 or ox1 >= fx2:
                        new_free.append((fx1, fx2))
                    else:
                        if fx1 < ox1:
                            new_free.append((fx1, ox1))
                        if ox2 < fx2:
                            new_free.append((ox2, fx2))
                free = new_free
        result[i] = free
    return result

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.25,
        classes=[2],
        verbose=False,
    )

    boxes_data = results[0].boxes
    free = {}

    if boxes_data is not None and boxes_data.id is not None:
        xyxy  = boxes_data.xyxy.cpu().numpy()
        ids   = boxes_data.id.cpu().numpy().astype(int)
        confs = boxes_data.conf.cpu().numpy()

        # Type checker: drop detections YOLO isn't confident are cars
        conf_mask = confs >= MIN_CONF
        xyxy  = xyxy[conf_mask]
        ids   = ids[conf_mask]

        # --- First pass: compute label for every detection ---
        labels = []
        draw_mask = []

        for box, car_id in zip(xyxy, ids):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            history = position_history[car_id]
            history.append((cx, cy))
            if len(history) > HISTORY_FRAMES:
                history.pop(0)

            if len(history) < 5:
                labels.append(None)
                draw_mask.append(False)
                continue

            mid         = len(history) // 2
            first_half  = history[:mid]
            second_half = history[mid:]
            med_first   = (np.median([p[0] for p in first_half]),
                           np.median([p[1] for p in first_half]))
            med_second  = (np.median([p[0] for p in second_half]),
                           np.median([p[1] for p in second_half]))
            displacement = np.hypot(
                med_second[0] - med_first[0],
                med_second[1] - med_first[1],
            )

            if displacement > MOVEMENT_THRESHOLD:
                moving_streak[car_id] += 1
            else:
                moving_streak[car_id] = 0

            is_moving = moving_streak[car_id] >= SUSTAIN_FRAMES
            labels.append("moving_car" if is_moving else "parked_car")
            draw_mask.append(True)

        labels    = np.array(labels, dtype=object)
        draw_mask = np.array(draw_mask, dtype=bool)  

        # Work only with detections that have enough history
        xyxy_d   = xyxy[draw_mask]
        ids_d    = ids[draw_mask]
        labels_d = labels[draw_mask]

        # --- Containment filter: only suppress if inside a LARGER box of the SAME type ---
        # Different types (moving vs parked) are real separate cars — never suppress them.
        if len(xyxy_d) > 1:
            cx_all = (xyxy_d[:, 0] + xyxy_d[:, 2]) / 2
            cy_all = (xyxy_d[:, 1] + xyxy_d[:, 3]) / 2
            areas  = (xyxy_d[:, 2] - xyxy_d[:, 0]) * (xyxy_d[:, 3] - xyxy_d[:, 1])

            inside = (
                (cx_all[:, None] > xyxy_d[None, :, 0]) &
                (cx_all[:, None] < xyxy_d[None, :, 2]) &
                (cy_all[:, None] > xyxy_d[None, :, 1]) &
                (cy_all[:, None] < xyxy_d[None, :, 3])
            )
            np.fill_diagonal(inside, False)

            same_type     = labels_d[:, None] == labels_d[None, :]
            inside_larger = inside & (areas[None, :] > areas[:, None]) & same_type
            keep          = ~inside_larger.any(axis=1)

            xyxy_d   = xyxy_d[keep]
            ids_d    = ids_d[keep]
            labels_d = labels_d[keep]

        # --- Draw ---
        parked_boxes = []
        for box, car_id, label in zip(xyxy_d, ids_d, labels_d):
            x1, y1, x2, y2 = box
            color = (0, 255, 255) if label == "moving_car" else (0, 0, 255)
            cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv.putText(
                frame,
                f"{label} #{car_id}",
                (int(x1), int(y1) - 8),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
            if label == "parked_car":
                parked_boxes.append((x1, y1, x2, y2))

        free = get_free_segments(SPOTS, parked_boxes)
        latest_free_map = free_segments_to_map(SPOTS, free)

    # --- Draw spot borders: green where free, red where occupied ---
    def rotated_rect(x1, y1, x2, y2, angle_deg):
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        angle = np.radians(angle_deg)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        rotated = np.array([
            [cx + cos_a * (px - cx) - sin_a * (py - cy),
             cy + sin_a * (px - cx) + cos_a * (py - cy)]
            for px, py in corners
        ], dtype=np.int32)
        return rotated.reshape((-1, 1, 2))

    SPOT_ANGLES = [-5]  # per-spot angles

    for i, (sx1, sy1, sx2, sy2) in enumerate(SPOTS):
        SPOT_ANGLE = SPOT_ANGLES[i]
        free_segs = free.get(i, [(sx1, sx2)])
        for (fx1, fx2) in free_segs:
            pts = rotated_rect(fx1, sy1, fx2, sy2, SPOT_ANGLE)
            cv.polylines(frame, [pts], True, (0, 255, 0), 2)
        occ_segs = [(sx1, sx2)]
        for (fx1, fx2) in free_segs:
            remaining = []
            for (ox1, ox2) in occ_segs:
                if fx2 <= ox1 or fx1 >= ox2:
                    remaining.append((ox1, ox2))
                else:
                    if ox1 < fx1:
                        remaining.append((ox1, fx1))
                    if fx2 < ox2:
                        remaining.append((fx2, ox2))
            occ_segs = remaining
        for (ox1, ox2) in occ_segs:
            pts = rotated_rect(ox1, sy1, ox2, sy2, SPOT_ANGLE)
            cv.polylines(frame, [pts], True, (0, 0, 255), 2)

    cv.imshow("Parking", frame)
    if cv.waitKey(frame_delay) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()