import numpy as np
import time
import cv2
from openvino.inference_engine import IENetwork, IECore


class GazeEstimation:
    """
    Class for the Gaze Pose Estimation Model.
    """

    def __init__(self, model_name, device, threshold=0.60):

        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.threshold = threshold
        self.net = None

        try:
            ie_core = IECore()
            self.model = ie_core.read_network(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Is it the correct model path?")

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

    def load_model(self):
        """
         This method is for loading the model to the device specified by the user.
         If your model requires any Plugins, this is where you can load them.
         """
        start = time.time()
        ie_core = IECore()
        self.net = ie_core.load_network(self.model, self.device, num_requests=4)
        print('[4] Loaded in  {:.3f} s'.format(time.time()-start))

    def predict(self, poses, left, right):
        """
         This method is meant for running predictions on the input image.
        """
        image_left = self.preprocess_input(left)
        image_right = self.preprocess_input(right)
        input_dict = {'left_eye_image': image_left, 'right_eye_image': image_right,
                      'head_pose_angles': poses}
        outputs = self.net.infer(input_dict)
        out = self.preprocess_outputs(outputs)

        return out

    def performance_counter(self, request_id):
        """
        Queries performance measures per layer to get feedback of what is the
        most time consuming layer.
        :param request_id: Index of Infer request value. Limited to device capabilities
        :return: Performance of the layer
        """
        perf_count = self.net.requests[request_id].get_perf_counts()
        return perf_count

    def preprocess_outputs(self, outputs):

        """
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        """

        return outputs[self.output_name]

    def preprocess_input(self, image):

        """
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        """

        image = cv2.resize(image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
        image = np.moveaxis(image, -1, 0)

        return image
