import numpy as np
import time
import cv2
from openvino.inference_engine import IENetwork, IECore


class PoseDetect:
    """
    Class for the Pose  Detection Model.
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
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

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
        print('Time for loading model', time.time() - start)

    def predict(self, image, w, h):
        """
         This method is meant for running predictions on the input image.
        """
        image_input = self.preprocess_input(image)
        input_dict = {self.input_name: image_input}
        start = time.time()

        outputs = self.net.infer(input_dict)
        coords = self.preprocess_outputs(outputs)

        print('y', outputs['angle_y_fc'])
        print('p', outputs['angle_p_fc'])
        print('r', outputs['angle_r_fc'])
        # coords, image = self.check_model(coords, image, w, h)

        return coords, image

    def check_model(self, coords, image, initial_w, initial_h):
        for obj in coords[0][0]:
            if obj[2] > self.threshold:
                xmin = int(obj[3] * initial_w)
                ymin = int(obj[4] * initial_h)
                xmax = int(obj[5] * initial_w)
                ymax = int(obj[6] * initial_h)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1.2)
        return coords, image

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
        n, c, h, w = self.input_shape
        assert (h == 60 and 60 == w)
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        image = np.moveaxis(image, -1, 0)
        return image


class Model_X:
    """
    Class for the Face Detection Model.
    """

    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        raise NotImplementedError

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        raise NotImplementedError

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        raise NotImplementedError

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        raise NotImplementedError

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        raise NotImplementedError
