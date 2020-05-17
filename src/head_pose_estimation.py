import numpy as np
import time
import cv2
from openvino.inference_engine import IENetwork, IECore


class HeadPoseEstimation:
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
            raise ValueError("Could not Initialise the network. Is it  the correct model path?")

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape

        self.output_names = ['angle_y_fc', 'angle_p_fc', 'angle_r_fc']
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
        print('[3] Loaded in  {:.3f} s'.format(time.time()-start))

    def predict(self, image):
        """
         This method is meant for running predictions on the input image.
        """
        image_input = self.preprocess_input(image)
        input_dict = {self.input_name: image_input}
        outputs = self.net.infer(input_dict)
        output = self.preprocess_outputs(outputs)

        return output

    def check_model(self, coords, image, initial_w, initial_h):
        pass

    def preprocess_outputs(self, outputs):

        """
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        """
        y = outputs[self.output_names[0]][0][0]
        p = outputs[self.output_names[1]][0][0]
        r = outputs[self.output_names[2]][0][0]
        output = np.array([y, p, r])
        return output

    def preprocess_input(self, image):

        """
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        """
        n, c, h, w = self.input_shape
        assert (h == 60 and 60 == w)
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        image = np.moveaxis(image, -1, 0)
        return image



