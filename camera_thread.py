import time

import cv2
import dv_processing as dv
import numpy as np
from PyQt5 import QtCore

# Filter Settings
CALIBRATION_DURATION = 2.0
SIGMA_THRESHOLD = 8.0
FILTER_BLUR_SIZE = (5, 5)
FILTER_THRESHOLD = 0.6


class CameraThread(QtCore.QThread):
    data_signal = QtCore.pyqtSignal(object, object)
    status_signal = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = True
        self.capture = None
        self.resolution = None

        self.filter_density_enabled = False

        self.hot_pixel_mask = None
        self.is_calibrating = False
        self.calibration_start = 0
        self.calibration_heatmap = None

    def toggle_density_filter(self):
        self.filter_density_enabled = not self.filter_density_enabled
        status = "ON" if self.filter_density_enabled else "OFF"
        self.status_signal.emit(f"Status: Density Filter {status}")

    def start_calibration(self):
        self.is_calibrating = True
        self.calibration_start = time.time()
        if self.resolution:
            self.calibration_heatmap = np.zeros(
                (self.resolution[1], self.resolution[0]), dtype=np.int32
            )
        self.status_signal.emit("Status: Calibrating... Cover lens.")

    def run(self):
        # Check if camera opens successfully
        try:
            self.capture = dv.io.camera.open()
        except RuntimeError:
            self.status_signal.emit("Error: Camera not found!")
            return

        self.resolution = self.capture.getEventResolution()
        w, h = self.resolution

        self.hot_pixel_mask = np.ones((h, w), dtype=bool)

        accumulator = dv.Accumulator(self.resolution)
        accumulator.setMinPotential(0.0)
        accumulator.setMaxPotential(1.0)
        accumulator.setDecayFunction(dv.Accumulator.Decay.STEP)

        while self.running:
            events = self.capture.getNextEventBatch()

            if events is not None and events.size() > 0:

                if self.is_calibrating:
                    ev_np = events.numpy()
                    np.add.at(self.calibration_heatmap, (ev_np["y"], ev_np["x"]), 1)
                    if time.time() - self.calibration_start > CALIBRATION_DURATION:
                        self._finalize_calibration()
                    continue

                ev_np = events.numpy()
                current_mask = self.hot_pixel_mask

                if self.filter_density_enabled:
                    heatmap = np.zeros((h, w), dtype=np.float32)
                    np.add.at(heatmap, (ev_np["y"], ev_np["x"]), 1.0)
                    blurred = cv2.GaussianBlur(heatmap, FILTER_BLUR_SIZE, 0)
                    density_mask = blurred > FILTER_THRESHOLD
                    current_mask = self.hot_pixel_mask & density_mask

                accumulator.accept(events)
                frame = accumulator.generateFrame()

                img_preview = None
                if frame is not None:
                    img_preview = frame.image
                    # Use 0 for black in uint8 image
                    img_preview[~current_mask] = 0

                valid_indices = current_mask[ev_np["y"], ev_np["x"]]
                ev_clean = ev_np[valid_indices]

                if len(ev_clean) > 0:
                    clean_batch = {
                        "x": ev_clean["x"],
                        "y": ev_clean["y"],
                        "timestamp": ev_clean["timestamp"],
                        "polarity": ev_clean["polarity"],
                    }
                    self.data_signal.emit(img_preview, clean_batch)

            else:
                time.sleep(0.001)

    def _finalize_calibration(self):
        if self.calibration_heatmap.max() == 0:
            self.status_signal.emit("Calibration Failed.")
            self.is_calibrating = False
            return

        mean_val = np.mean(self.calibration_heatmap)
        std_val = np.std(self.calibration_heatmap)
        cutoff = mean_val + (SIGMA_THRESHOLD * std_val)

        self.hot_pixel_mask = self.calibration_heatmap < cutoff
        count = np.sum(~self.hot_pixel_mask)
        self.status_signal.emit(f"Active: Masked {count} hot pixels.")
        self.is_calibrating = False

    def stop(self):
        self.running = False
        self.wait()
