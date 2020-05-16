import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
from input_feeder import InputFeeder
from model import PoseDetect, LandMarksDetect, GazeDetect, FaceDetect
from mouse_controller import MouseController
import datetime

# constants
MAIN_WINDOW_NAME = 'INTEL EDGE AI'


def main(args):
    model = args.model
    device = args.device
    video_file = args.video
    input_type = args.input_type
    output_path = args.output_path
    toggle = args.toggle
    stats = args.stats

    if stats == 'true':
        stats = True
    else:
        stats = False

    if toggle == 'true':
        toggle = True
    else:
        toggle = False

    start_model_load_time = time.time()
    start_time = datetime.datetime.now().strftime("%H:%M:%S")

    m = FaceDetect(model, device)
    m.load_model()
    land = LandMarksDetect('./resources/models/landmarks-regression-retail-0009/FP16/landmarks-regression-retail'
                           '-0009',
                           device)
    land.load_model()
    pose = PoseDetect('./resources/models/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001', device)
    pose.load_model()

    gaze = GazeDetect('./resources/models/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002', device)
    gaze.load_model()

    total_model_load_time = time.time() - start_model_load_time
    print(f'Time taken to load model is = {total_model_load_time}')

    try:
        feed = InputFeeder(input_type=input_type, input_file=video_file)
        feed.load_data()
        initial_w = int(feed.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        initial_h = int(feed.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(feed.cap.get(cv2.CAP_PROP_FPS))
        counter = 0
        mouse = MouseController('medium', 'fast')
        cv2.namedWindow(MAIN_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

        average_inf_time = 0

        start_inference = time.time()

        for frame, _ in feed.next_batch():
            if not _:
                break
            try:
                counter += 1
                coord = m.predict(frame, (initial_w, initial_h))
                #m.requests[0].get_perf_counts()
                for i in range(len(coord)):
                    xmin, ymin, xmax, ymax = coord[i]
                    cropped_image = frame[ymin:ymax, xmin:xmax]
                    cropped_left, cropped_right = land.predict(cropped_image)
                    if cropped_right.shape[0] < 60 or cropped_left.shape[1] < 60:
                        break
                    if cropped_right.shape[1] < 60 or cropped_left.shape[0] < 60:
                        break
                    poses = pose.predict(cropped_image)
                    gz = gaze.predict(poses, cropped_left, cropped_right)
                    mouse.move(gz[0][0], gz[0][1])

                    det_time = time.time() - start_inference
                    inf_time_message = "Inference time: {:.3f}ms" \
                        .format(det_time * 1000)

                    cv2.putText(frame, inf_time_message, (15, 35),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

                    cv2.putText(frame,
                                'Inference Running  %d FPs' % round(fps, 1),
                                (15, 65), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

                    # If user pass statistics argument to true
                    y_max = 70
                    if stats:
                        cv2.putText(frame,
                                    'Inference Running Time: %0.3f s' % (time.time() - start_inference),
                                    (10, initial_h - y_max), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        y_max += 25
                        cv2.putText(frame,
                                    'Pre-processing Running  Time: %0.3f s' % total_model_load_time,
                                    (10, initial_h - y_max), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        y_max += 35
                        cv2.putText(frame,
                                    'STATISTICS :',
                                    (10, initial_h - y_max), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

                # print(f'Total Time taken for inference is {time.time() - start_inference_time}')
                if not toggle:
                    cv2.imshow(MAIN_WINDOW_NAME, frame)
                else:
                    frame = np.zeros((480, 680))
                    cv2.putText(frame,
                                'STATISTICS :',
                                (20, y_max), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    y_max += 25
                    cv2.putText(frame,
                                'Pre-processing Running  Time: %0.3f s' % total_model_load_time,
                                (40, y_max), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_max += 25

                    cv2.putText(frame,
                                'Inference Running Time: %0.3f s' % (time.time() - start_inference),
                                (40, y_max), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.imshow(MAIN_WINDOW_NAME, frame)

                cv2.waitKey(1)
            except Exception as e:
                print('Could not run Inference', e)
        average_inf_time = average_inf_time / counter
        with open(os.path.join(output_path, 'inference.txt'), 'w') as f:
            f.write('Inference Time Average : ' + str(average_inf_time) + '\n')
            f.write('Frame Per Second  : ' + str(fps) + '\n')
            f.write('Model load time  : ' + str(total_model_load_time) + '\n')

        feed.close()
    except Exception as e:
        print("Could not run Inference: ", e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, help='Path to model files', type=str)
    parser.add_argument('-d', '--device', default='CPU', help='Specify the target device to infer on',
                        type=str, required=False)
    parser.add_argument('-v', '--video', default=None, help='Specify the video path, not required for camera ')
    parser.add_argument('-o', '--output_path', default='./resources/results')
    parser.add_argument('-i', '--input_type', default='video', help='Specify the input pipeline video or camera input')
    parser.add_argument('-t', '--toggle', default='false', help='Toggle the camera when enable', type=str)
    parser.add_argument('-s', '--stats', default='false', type=str, help='Enable statistics')

    args = parser.parse_args()
    # main(args)
    main(args)

# TODO in this file


# - Experiment with different model precision
# - use get_performance_count api  for running time of each of the models
# - use VTune Amplifier and update README



# TODO
# Toggle camera and show stats only : (no show just a window with stats)
#       Create window without video cv2
#       Print statistics and see performance

# ---------------------------
# Pre processing pipeline for both video and camera(command line argument)(ok)
# - user argument running time pre-processing and inference pipeline(video, cam)(stats)(ok)
# - Print the running time stats of application:(ok)


# How to run requirement.txt python
# Benchmark openvino how to ?
# Links to documentation
# Opencv : how to create window and move to center


source env/bin/activate
pip install -r requirements.txt