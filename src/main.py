import time
import json
import os
import cv2
import argparse
from input_feeder import InputFeeder
from face_detection import FaceDetection
from facial_landmarks_detection import LandMarksDetection
from head_pose_estimation import HeadPoseEstimation
from gaze_estimation import GazeEstimation
from mouse_controller import MouseController

# constants
MAIN_WINDOW_NAME = 'INTEL EDGE AI'


def parse_models_file(label, path):
    """
        Parses the model file.
        Reads models
    """
    assert os.path.isfile(path), "{} file doesn't exist".format(path)
    m = json.loads(open(path).read())
    model = ''
    for idx, item in enumerate(m['models']):
        if label == item['label']:
            model = item['model']
            break

    return model


def performance_counts(perf_count):
    """
    print information about layers of the model.

    :param perf_count: Dictionary consists of status of the layers.
    :return: None
    """
    print("{:<70} {:<15} {:<15} {:<15} {:<10}".format('name', 'layer_type',
                                                      'exec_type', 'status',
                                                      'real_time, us'))
    for layer, stats in perf_count.items():
        print("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer,
                                                          stats['layer_type'],
                                                          stats['exec_type'],
                                                          stats['status'],
                                                          stats['real_time']))


def main(args):
    device = args.device
    video_file = args.video
    input_type = args.input_type
    output_path = args.output_path
    toggle = args.toggle
    stats = args.stats
    model = args.model

    if stats == 'true':
        stats = True
    else:
        stats = False

    if toggle == 'true':
        toggle = True
    else:
        toggle = False

    # Start Model Loading
    start_model_load_time = time.time()
    print(f'[INFO] Started Model Loading...........')

    face_model = FaceDetection(parse_models_file(
        label='face_detection', path=model),
        device)
    face_model.load_model()

    # Load Landmark model
    landmark_model = LandMarksDetection(
        parse_models_file(label='facial_landmarks_detection', path=model),
        device)
    landmark_model.load_model()
    pose_estimation_model = HeadPoseEstimation(
        parse_models_file(label='head_pose_estimation', path=model),
        device)
    pose_estimation_model.load_model()

    gaze_estimation_model = GazeEstimation(
        parse_models_file(label='gaze_estimation', path=model), device)
    gaze_estimation_model.load_model()

    total_model_load_time = time.time() - start_model_load_time
    print('[TOTAL] Loaded in {:.3f} ms'.format(total_model_load_time))

    # End Model Loading

    try:
        feed = InputFeeder(input_type=input_type, input_file=video_file)
        feed.load_data()
        initial_w = int(feed.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        initial_h = int(feed.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(feed.cap.get(cv2.CAP_PROP_FPS))
        counter = 0
        mouse = MouseController('medium', 'fast')
        if not toggle:
            cv2.namedWindow(MAIN_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        average_inf_time = 0
        for frame, _ in feed.next_batch():
            if not _:
                break
            try:
                counter += 1
                # Start Inferences
                coord = face_model.predict(frame, (initial_w, initial_h))

                for i in range(len(coord)):
                    xmin, ymin, xmax, ymax = coord[i]
                    cropped_image = frame[ymin:ymax, xmin:xmax]
                    # Landmark Inference
                    cropped_left, cropped_right = landmark_model.predict(cropped_image)
                    if cropped_right.shape[0] < 60 or cropped_left.shape[1] < 60:
                        break
                    if cropped_right.shape[1] < 60 or cropped_left.shape[0] < 60:
                        break
                    # Pose Estimation Inference
                    poses = pose_estimation_model.predict(cropped_image)
                    # Gaze Estimation Inference
                    gz = gaze_estimation_model.predict(poses, cropped_left, cropped_right)
                    # Mouse Controller
                    mouse.move(gz[0][0], gz[0][1])
                    # If user pass statistics argument to true
                    if stats:
                        # Print performance
                        performance_counts(
                            face_model.performance_counter(0)
                        )
                        performance_counts(
                            pose_estimation_model.performance_counter(0)
                        )
                        performance_counts(
                            landmark_model.performance_counter(0)
                        )
                        performance_counts(
                            gaze_estimation_model.performance_counter(0)
                        )

                if not toggle:
                    # Output Camera or Video
                    cv2.imshow(MAIN_WINDOW_NAME, frame)

                else:
                    # Print Statistics only no camera or video
                    performance_counts(
                        face_model.performance_counter(0)
                    )
                    performance_counts(
                        pose_estimation_model.performance_counter(0)
                    )
                    performance_counts(
                        landmark_model.performance_counter(0)
                    )
                    performance_counts(
                        gaze_estimation_model.performance_counter(0)
                    )

                cv2.waitKey(1)
            except Exception as e:
                print('Could not run Inference', e)


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
    main(args)

# Benchmarks
# Stats output
#
