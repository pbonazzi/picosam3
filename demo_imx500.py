"""
PicoSAM3 student — interactive IMX500 segmentation demo.

Controls:
  • Left-click + drag  : draw a new ROI bounding box prompt
  • r                  : reset ROI to full frame
  • ESC / q            : quit
"""

import time
import os
import cv2
import numpy as np
from picamera2 import Picamera2, CompletedRequest
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics

MODEL_PATH   = os.path.join(os.path.dirname(__file__),
                             "checkpoints", "rpk", "network.rpk")
WINDOW_NAME  = "PicoSAM3 IMX500 Segmentation"
DISPLAY_SIZE = (1280, 960)          # W × H — camera main stream is set to this
SENSOR_W, SENSOR_H = 4056, 3040

imx500  = None
picam2  = None

latest_mask  = None
latest_frame = None
last_request = None

roi_x, roi_y = 0, 0
roi_w, roi_h = SENSOR_W, SENSOR_H

dragging    = False
drag_start  = (0, 0)
drag_end    = (0, 0)
drawn_rect  = None   # (x0,y0,x1,y1) in display px — set after user finishes drag


def apply_roi():
    rx = max(0, min(roi_x, SENSOR_W - roi_w))
    ry = max(0, min(roi_y, SENSOR_H - roi_h))
    rw = max(64, min(roi_w, SENSOR_W))
    rh = max(64, min(roi_h, SENSOR_H))
    imx500.set_inference_roi_abs((rx, ry, rw, rh))


def mouse_callback(event, x, y, flags, param):
    global dragging, drag_start, drag_end
    global roi_x, roi_y, roi_w, roi_h, drawn_rect

    if event == cv2.EVENT_LBUTTONDOWN:
        dragging   = True
        drag_start = (x, y)
        drag_end   = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        drag_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        drag_end = (x, y)

        x0, x1 = sorted([drag_start[0], drag_end[0]])
        y0, y1 = sorted([drag_start[1], drag_end[1]])

        # Convert display → sensor coordinates
        sx = int(x0 * SENSOR_W / DISPLAY_SIZE[0])
        sy = int(y0 * SENSOR_H / DISPLAY_SIZE[1])
        sw = int((x1 - x0) * SENSOR_W / DISPLAY_SIZE[0])
        sh = int((y1 - y0) * SENSOR_H / DISPLAY_SIZE[1])

        if sw >= 64 and sh >= 64:
            roi_x, roi_y, roi_w, roi_h = sx, sy, sw, sh
            drawn_rect = (x0, y0, x1, y1)   # store display-space rect
            apply_roi()


def segmentation_callback(request: CompletedRequest):
    global latest_mask, latest_frame, last_request

    frame = request.make_array("main")
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Camera main stream is configured to DISPLAY_SIZE — no resize needed
    latest_frame = frame
    last_request = request

    outputs = imx500.get_outputs(request.get_metadata())
    if outputs is None:
        return

    raw  = np.squeeze(outputs[0])
    latest_mask = (raw > 0).astype(np.uint8) * 255


if __name__ == "__main__":
    # Materialise Qt window handle before camera threads start
    cv2.namedWindow(WINDOW_NAME)
    cv2.imshow(WINDOW_NAME, np.zeros((DISPLAY_SIZE[1], DISPLAY_SIZE[0], 3), np.uint8))
    cv2.waitKey(1)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    imx500 = IMX500(MODEL_PATH)
    intr   = imx500.network_intrinsics or NetworkIntrinsics()
    intr.task = "segmentation"
    intr.update_with_defaults()

    picam2 = Picamera2(imx500.camera_num)
    # Ask the ISP to deliver frames at exactly DISPLAY_SIZE so get_roi_scaled
    # returns display-space coordinates with no further scaling needed.
    config = picam2.create_preview_configuration(
        main={"size": DISPLAY_SIZE, "format": "RGB888"},
        controls={"FrameRate": intr.inference_rate},
        buffer_count=8,
    )
    picam2.start(config, show_preview=False)
    picam2.pre_callback = segmentation_callback

    apply_roi()
    print("PicoSAM3 IMX500 demo running.")
    print("  Left-click + drag to set ROI  |  r = reset  |  ESC/q = quit")

    try:
        while True:
            if latest_frame is None:
                time.sleep(0.005)
                continue

            frame   = latest_frame.copy()
            request = last_request

            if latest_mask is not None and request is not None:
                # get_roi_scaled returns coords in the main-stream frame space
                # (= DISPLAY_SIZE because that's what we configured)
                rx, ry, rw, rh = imx500.get_roi_scaled(request)

                mask_disp = cv2.resize(latest_mask, (rw, rh),
                                       interpolation=cv2.INTER_NEAREST)
                roi_crop = frame[ry:ry+rh, rx:rx+rw].astype(np.float32)
                roi_crop[..., 0] = np.clip(
                    roi_crop[..., 0] + mask_disp.astype(np.float32) * 0.5,
                    0, 255,
                )
                frame[ry:ry+rh, rx:rx+rw] = roi_crop.astype(np.uint8)
            else:
                cv2.putText(frame, "Loading model on IMX500...",
                            (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

            # Green box — only after user draws one, in display coordinates
            if drawn_rect is not None:
                x0, y0, x1, y1 = drawn_rect
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

            # Cyan preview while dragging
            if dragging:
                cv2.rectangle(frame, drag_start, drag_end, (0, 255, 255), 2)

            cv2.putText(frame, "PicoSAM3 | drag to set ROI | r=reset | q=quit",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            elif key == ord('r'):
                roi_x, roi_y, roi_w, roi_h = 0, 0, SENSOR_W, SENSOR_H
                drawn_rect = None
                apply_roi()

    finally:
        picam2.stop()
        cv2.destroyAllWindows()
