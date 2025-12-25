# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import collections
from functools import partial
import re
import time

import numpy as np
from PIL import Image
import svgwrite
import gstreamer
import math

from pose_engine import PoseEngine
from pose_engine import KeypointType

import serial
ser = serial.Serial('/dev/ttyCH341USB0', 9600)


# Global variable for tracking
tracked_face = None
switch_threshold = 1.4  # 40% larger to trigger switch

last_send_time = 0
send_interval = 0.1  # seconds 
headless_flag = False


EDGES = (
    (KeypointType.NOSE, KeypointType.LEFT_EYE),
    (KeypointType.NOSE, KeypointType.RIGHT_EYE),
    (KeypointType.NOSE, KeypointType.LEFT_EAR),
    (KeypointType.NOSE, KeypointType.RIGHT_EAR),
    (KeypointType.LEFT_EAR, KeypointType.LEFT_EYE),
    (KeypointType.RIGHT_EAR, KeypointType.RIGHT_EYE),
    (KeypointType.LEFT_EYE, KeypointType.RIGHT_EYE),
    (KeypointType.LEFT_SHOULDER, KeypointType.RIGHT_SHOULDER),
    (KeypointType.LEFT_SHOULDER, KeypointType.LEFT_ELBOW),
    (KeypointType.LEFT_SHOULDER, KeypointType.LEFT_HIP),
    (KeypointType.RIGHT_SHOULDER, KeypointType.RIGHT_ELBOW),
    (KeypointType.RIGHT_SHOULDER, KeypointType.RIGHT_HIP),
    (KeypointType.LEFT_ELBOW, KeypointType.LEFT_WRIST),
    (KeypointType.RIGHT_ELBOW, KeypointType.RIGHT_WRIST),
    (KeypointType.LEFT_HIP, KeypointType.RIGHT_HIP),
    (KeypointType.LEFT_HIP, KeypointType.LEFT_KNEE),
    (KeypointType.RIGHT_HIP, KeypointType.RIGHT_KNEE),
    (KeypointType.LEFT_KNEE, KeypointType.LEFT_ANKLE),
    (KeypointType.RIGHT_KNEE, KeypointType.RIGHT_ANKLE),
)


def shadow_text(dwg, x, y, text, font_size=16):
    dwg.add(dwg.text(text, insert=(x + 1, y + 1), fill='black',
                     font_size=font_size, style='font-family:sans-serif'))
    dwg.add(dwg.text(text, insert=(x, y), fill='white',
                     font_size=font_size, style='font-family:sans-serif'))


def draw_pose(dwg, pose, src_size, inference_box, color='yellow', threshold=0.2):
    box_x, box_y, box_w, box_h = inference_box
    scale_x, scale_y = src_size[0] / box_w, src_size[1] / box_h
    xys = {}
    for label, keypoint in pose.keypoints.items():
        if keypoint.score < threshold: continue
        # Offset and scale to source coordinate space.
        kp_x = int((keypoint.point[0] - box_x) * scale_x)
        kp_y = int((keypoint.point[1] - box_y) * scale_y)

        xys[label] = (kp_x, kp_y)
        dwg.add(dwg.circle(center=(int(kp_x), int(kp_y)), r=5,
                           fill='cyan', fill_opacity=keypoint.score, stroke=color))

    for a, b in EDGES:
        if a not in xys or b not in xys: continue
        ax, ay = xys[a]
        bx, by = xys[b]
        dwg.add(dwg.line(start=(ax, ay), end=(bx, by), stroke=color, stroke_width=2))


def avg_fps_counter(window_size):
    window = collections.deque(maxlen=window_size)
    prev = time.monotonic()
    yield 0.0  # First fps value.

    while True:
        curr = time.monotonic()
        window.append(curr - prev)
        prev = curr
        yield len(window) / sum(window)


def run(inf_callback, render_callback):
    global headless_flag
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mirror', help='flip video horizontally', action='store_true')
    parser.add_argument('--model', help='.tflite model path.', required=False)
    parser.add_argument('--res', help='Resolution', default='640x480',
                        choices=['480x360', '640x480', '1280x720'])
    parser.add_argument('--videosrc', help='Which video source to use', default='/dev/video1')
    parser.add_argument('--h264', help='Use video/x-h264 input', action='store_true')
    parser.add_argument('--jpeg', help='Use image/jpeg input', action='store_true')
    parser.add_argument('--headless', help='Disable display output', action='store_true')

    args = parser.parse_args()

    default_model = 'models/mobilenet/posenet_mobilenet_v1_075_%d_%d_quant_decoder_edgetpu.tflite'
    if args.res == '480x360':
        src_size = (640, 480)
        appsink_size = (480, 360)
        model = args.model or default_model % (353, 481)
    elif args.res == '640x480':
        src_size = (640, 480)
        appsink_size = (640, 480)
        model = args.model or default_model % (481, 641)
    elif args.res == '1280x720':
        src_size = (1280, 720)
        appsink_size = (1280, 720)
        model = args.model or default_model % (721, 1281)

    if args.headless:
        headless_flag = True

    print('Loading model: ', model)
    engine = PoseEngine(model)
    input_shape = engine.get_input_tensor_shape()
    inference_size = (input_shape[2], input_shape[1])

    gstreamer.run_pipeline(partial(inf_callback, engine), partial(render_callback, engine),
                           src_size, inference_size,
                           mirror=args.mirror,
                           videosrc=args.videosrc,
                           h264=args.h264,
                           jpeg=args.jpeg
                           )
    


def main():
    n = 0
    sum_process_time = 0
    sum_inference_time = 0
    ctr = 0
    fps_counter = avg_fps_counter(30)



    def run_inference(engine, input_tensor):
        return engine.run_inference(input_tensor)

    def center_of_rect(rect):
        x, y, w, h = rect
        return (x + w // 2, y + h // 2)

    def euclidean_distance(p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5

    def render_overlay(engine, output, src_size, inference_box):
        global tracked_face, last_send_time, headless_flag
        svg_canvas = svgwrite.Drawing('', size=src_size)
        outputs, inference_time = engine.ParseOutput()

        shadow_text(svg_canvas, 10, 20,
            f"Poses: {len(outputs)}   Inference: {inference_time * 1000:.1f}ms")

        #face_kps = [KeypointType.NOSE, KeypointType.LEFT_EYE, KeypointType.RIGHT_EYE, KeypointType.LEFT_EAR, KeypointType.RIGHT_EAR]
        face_kps = [KeypointType.NOSE, KeypointType.LEFT_EYE, KeypointType.RIGHT_EYE]

        best_face = None
        best_area = 0 

        
        # Camera FOVs — you can adjust these
        HFOV = 69  # degrees
        VFOV = 49  # degrees

        frame_center = (src_size[0] // 2, src_size[1] // 2)

        for pose in outputs:
            face_points = []
            for kp_type in face_kps:
                kp = pose.keypoints.get(kp_type)
                if kp and kp.score > 0.3:
                    box_x, box_y, box_w, box_h = inference_box
                    scale_x = src_size[0] / box_w
                    scale_y = src_size[1] / box_h
                    x = int((kp.point[0] - box_x) * scale_x)
                    y = int((kp.point[1] - box_y) * scale_y)
                    face_points.append((x, y))

            if len(face_points) >= 3:
                xs, ys = zip(*face_points)
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                w, h = x_max - x_min, y_max - y_min
                area = w * h
                center = (x_min + w // 2, y_min + h // 2)
                face_rect = (x_min, y_min, w, h)

                # Draw all faces
                color = 'green'
                if tracked_face:
                    t_x, t_y, t_w, t_h = tracked_face
                    t_area = t_w * t_h
                    t_center = center_of_rect(tracked_face)
                    dist = euclidean_distance(center, t_center)

                    # Same face if area similar and close
                    if 0.7 * t_area < area < 1.3 * t_area and dist < 100:
                        best_face = face_rect
                        best_area = area
                        color = 'red'
                    elif area > t_area * switch_threshold:  # Change face
                        best_face = face_rect
                        best_area = area
                        color = 'red'
                else: 
                    if area > best_area:
                        best_face = face_rect
                        best_area = area

                svg_canvas.add(svg_canvas.rect(insert=(x_min, y_min), size=(w, h), stroke=color, fill='none', stroke_width=2))

            draw_pose(svg_canvas, pose, src_size, inference_box)

        # Draw and send offset for the best face
        if best_face:
            x, y, w, h = best_face
            cx, cy = center_of_rect(best_face)
            svg_canvas.add(svg_canvas.circle(center=(cx, cy), r=5, fill='blue'))

            # Calculate deviasion from the center of the face to the center of the frame
            dx = cx - frame_center[0]
            dy = cy - frame_center[1]

            
            # Normalized offsets [-1, 1]
            norm_dx = dx / (src_size[0] / 2)
            norm_dy = dy / (src_size[1] / 2)
            norm_dx = max(-1, min(1,norm_dx))
            norm_dy = max(-1, min(1,norm_dy))



            # Calculate angle offsets
            angle_x = norm_dx * (HFOV / 2)
            angle_y = norm_dy * (VFOV / 2)


            # Check if it is time to send message
            now = time.monotonic()
            if now - last_send_time > send_interval:
                last_send_time = now

                # Send only if angle is grater than 1 deg
                if abs(angle_x) > 1 or abs(angle_y) > 1:
                    msg = f"{angle_x},{angle_y}\n"

                    try:
                        ser.write(msg.encode('utf-8'))
                        ser.flush()
                    except:
                        print("Warning: Could not write to serial.")

            # Track the best face
            tracked_face = best_face
        else:
            tracked_face = None


        # In order to activate it you must pass it from the terminal as a flag --headless.
        # It doesn't show the squares and keypoints in the video feed.
        if headless_flag:
            return "", False
        else:
            return svg_canvas.tostring(), False
        

        

    run(run_inference, render_overlay)


if __name__ == '__main__':
    main()
