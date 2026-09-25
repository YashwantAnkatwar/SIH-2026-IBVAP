#!/usr/bin/env python3
"""
create_indian_checkpoint_feed.py

Generates an authentic Indian Border Checkpoint video feed (BOP-03)
using real Indian vehicles and license plates from the State-wise dataset.
Simulates vehicles approaching a Border Outpost (BOP) sentry gate,
pausing for inspection, and proceeding through the barrier.
"""

import os
import glob
import xml.etree.ElementTree as ET
import cv2
import numpy as np

OUTPUT_PATH = "videos/bop3_indian_checkpoint_traffic.mp4"
WIDTH = 960
HEIGHT = 540
FPS = 25
DURATION_PER_VEHICLE_SEC = 6.0
FRAMES_PER_VEHICLE = int(FPS * DURATION_PER_VEHICLE_SEC)

# Target border and high-priority state samples
STATES_PRIORITY = ["JK", "PB", "DL", "HR", "MH", "GJ", "UP", "KA"]


def find_best_vehicle_crops():
    samples = []
    base_dir = "data/indian_plates/State-wise_OLX"
    for state in STATES_PRIORITY:
        state_dir = os.path.join(base_dir, state)
        if not os.path.isdir(state_dir):
            continue
        xmls = sorted(glob.glob(os.path.join(state_dir, "*.xml")))
        for xml_file in xmls[:4]:
            jpg_file = xml_file.replace(".xml", ".jpg")
            if not os.path.exists(jpg_file):
                continue
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                plate_text = root.find("object/name").text
                bnd = root.find("object/bndbox")
                xmin = int(bnd.find("xmin").text)
                ymin = int(bnd.find("ymin").text)
                xmax = int(bnd.find("xmax").text)
                ymax = int(bnd.find("ymax").text)
                
                img = cv2.imread(jpg_file)
                if img is not None and (xmax - xmin) >= 35:
                    samples.append({
                        "state": state,
                        "plate": plate_text,
                        "image": img,
                        "plate_box": (xmin, ymin, xmax, ymax)
                    })
            except Exception:
                continue
    return samples


def draw_checkpoint_background():
    # Base asphalt road
    bg = np.full((HEIGHT, WIDTH, 3), (45, 45, 50), dtype=np.uint8)

    # Road shoulder / border dirt terrain
    bg[0:HEIGHT, 0:180] = (60, 80, 100)      # Sandy / gravel left shoulder
    bg[0:HEIGHT, 780:WIDTH] = (60, 80, 100)  # Sandy / gravel right shoulder

    # Road lane markings
    cv2.line(bg, (180, 0), (180, HEIGHT), (255, 255, 255), 3)
    cv2.line(bg, (780, 0), (780, HEIGHT), (255, 255, 255), 3)

    # Dashed center line
    for y in range(0, HEIGHT, 40):
        cv2.line(bg, (480, y), (480, y + 20), (255, 255, 255), 3)

    # Checkpoint Inspection Stop Line (Yellow & Black stripes)
    stop_y = int(HEIGHT * 0.65)
    cv2.rectangle(bg, (180, stop_y), (780, stop_y + 16), (30, 200, 240), -1)
    for sx in range(180, 780, 40):
        cv2.line(bg, (sx, stop_y), (sx + 20, stop_y + 16), (20, 20, 20), 4)

    # Sentry Post Bunker Booth on the right
    cv2.rectangle(bg, (785, stop_y - 120), (940, stop_y + 40), (70, 85, 60), -1)  # Military green booth
    cv2.rectangle(bg, (795, stop_y - 105), (930, stop_y - 45), (180, 210, 210), -1) # Window
    cv2.putText(bg, "BOP CHECKPOST 03", (790, stop_y - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

    # Boom Barrier Gate
    cv2.line(bg, (785, stop_y), (460, stop_y), (0, 0, 220), 8) # Red boom arm
    cv2.line(bg, (460, stop_y), (180, stop_y), (255, 255, 255), 6) # White boom arm

    return bg


def generate_video():
    samples = find_best_vehicle_crops()
    if not samples:
        print("ERROR: No Indian vehicle samples found.")
        return False

    print(f"Found {len(samples)} Indian vehicle samples. Generating video: {OUTPUT_PATH}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (WIDTH, HEIGHT))

    bg = draw_checkpoint_background()

    for idx, sample in enumerate(samples[:10]):
        veh_img = sample["image"]
        state = sample["state"]
        plate_gt = sample["plate"]
        vh, vw = veh_img.shape[:2]

        print(f"  Rendering Vehicle #{idx+1} [{state}] {plate_gt}...")

        # Target dimensions in checkpoint lane
        target_w = 420
        target_h = int(vh * (target_w / vw))
        if target_h > 360:
            target_h = 360
            target_w = int(vw * (target_h / vh))

        resized_veh = cv2.resize(veh_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Path: Approaches from y=-target_h, stops at inspection line, then drives past
        # Stop position center: x = 480 (centered on road), y = stop_y - target_h + 30
        stop_y = int(HEIGHT * 0.65)
        stop_target_y = stop_y - target_h + 40
        start_y = -target_h
        exit_y = HEIGHT + 20

        # Timing phases within FRAMES_PER_VEHICLE (150 frames):
        # 0..45: Approach (drive from start_y to stop_target_y)
        # 46..105: Stopped at inspection line (60 frames = 2.4s pause for ANPR read)
        # 106..149: Exit (drive from stop_target_y to exit_y)
        for f in range(FRAMES_PER_VEHICLE):
            frame = bg.copy()

            if f < 45:
                progress = f / 45.0
                curr_y = int(start_y + (stop_target_y - start_y) * (1 - (1 - progress)**2))
            elif f <= 105:
                curr_y = stop_target_y
            else:
                progress = (f - 105) / 44.0
                curr_y = int(stop_target_y + (exit_y - stop_target_y) * (progress**2))

            curr_x = int((WIDTH - target_w) / 2)

            # Overlay vehicle onto road
            # Calculate overlapping bounding box
            y1_frame = max(0, curr_y)
            y2_frame = min(HEIGHT, curr_y + target_h)
            x1_frame = max(0, curr_x)
            x2_frame = min(WIDTH, curr_x + target_w)

            y1_veh = max(0, -curr_y)
            y2_veh = y1_veh + (y2_frame - y1_frame)
            x1_veh = max(0, -curr_x)
            x2_veh = x1_veh + (x2_frame - x1_frame)

            if y2_frame > y1_frame and x2_frame > x1_frame:
                frame[y1_frame:y2_frame, x1_frame:x2_frame] = resized_veh[y1_veh:y2_veh, x1_veh:x2_veh]

            # Tactical HUD Stamp
            cv2.putText(frame, "CAM: BOP-03 [VEHICLE CHECKPOST]", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            cv2.putText(frame, "SECTOR: JAMMU-PUNJAB HIGHWAY", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            out.write(frame)

    out.release()
    print(f"SUCCESS: Generated {OUTPUT_PATH} with {len(samples[:10]) * FRAMES_PER_VEHICLE} frames.")
    return True


if __name__ == "__main__":
    generate_video()
