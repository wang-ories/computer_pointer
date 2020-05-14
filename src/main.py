# Import

import cv2
import argparse
from input_feeder import InputFeeder
from model import PoseDetect


def main(args):
    model = args.model
    device = args.device
    video_file = args.video
    threshold = args.threshold

    fd = PoseDetect(model, device, threshold)
    fd.load_model()

    # Feed
    feed = InputFeeder(input_type='video', input_file=video_file)
    feed.load_data()

    initial_w = int(feed.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(feed.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    for frame in feed.next_batch():
        fd.predict(frame, initial_w, initial_h)

        # Real time output
        cv2.waitKey(1)
        cv2.imshow("PC SYSTEM 0.1", frame)

    feed.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--video', default=None)
    parser.add_argument('--output_path', default='/results')
    parser.add_argument('--threshold', default=0.60)

    args = parser.parse_args()

    main(args)
