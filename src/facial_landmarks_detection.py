import numpy as np
import time
import cv2
from openvino.inference_engine import IENetwork, IECore


class LandMarksDetection:
    """
    Class for the Pose  Detection Model.
    """

    def __init__(self, model_name, device):

        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
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
        print('[2] Loaded in  {:.3f} s'.format(time.time() - start))

    def draw_output(self, image, coord, depth=2, pos='left'):
        """
        Draw output for left and right eyes
        """
        x, y = coord
        initial_h, initial_w, n = image.shape
        x_min = int(x * initial_w) - depth
        x_max = int(x * initial_w) + depth
        y_min = int(y * initial_h) - depth
        y_max = int(y * initial_h) + depth
        # cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 255, 255), 1)

        img = image[y_min:y_max, x_min:x_max]

        return img

    def predict(self, image):
        """
         This method is meant for running predictions on the input image.
        """
        image_input = self.preprocess_input(image)
        input_dict = {self.input_name: image_input}
        start=time.time()

        outputs = self.net.infer(input_dict)
        out = self.preprocess_outputs(outputs)

        # Get the four first coordinates for left and right eyes
        x0 = out[0][0]
        y0 = out[1][0]
        x1 = out[2][0]
        y1 = out[3][0]

        #
        left_eye = (x0[0], y0[0])
        right_eye = (x1[0], y1[0])

        # draw the left eye
        left = self.draw_output(image, left_eye, 30)
        # draw the right eye
        right = self.draw_output(image, right_eye, 30)

        return left, right

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
        return outputs[self.output_name][0]

    def preprocess_input(self, image):

        """
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        """
        n, c, h, w = self.input_shape
        assert (h == 48 and 48 == w)

        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        image = np.moveaxis(image, -1, 0)
        return image
