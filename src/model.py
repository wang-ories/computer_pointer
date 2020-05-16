import numpy as np
import time
import cv2
from openvino.inference_engine import IENetwork, IECore


class PoseDetect:
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

        self.net = ie_core.load_network(self.model, self.device)
        print('Time for loading model', time.time() - start)

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


class LandMarksDetect:
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
        self.net = ie_core.load_network(self.model, self.device)
        # print('Time for loading model', time.time() - start)

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
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 255, 255), 1)

        img = image[y_min:y_max, x_min:x_max]

        return img

    def predict(self, image):
        """
         This method is meant for running predictions on the input image.
        """
        image_input = self.preprocess_input(image)
        input_dict = {self.input_name: image_input}
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

    def check_model(self):
        pass

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


class GazeDetect:
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
        ie_core = IECore()
        self.net = ie_core.load_network(self.model, self.device)

    def predict(self, poses, left, right):
        """
         This method is meant for running predictions on the input image.
        """

        image_left = self.preprocess_input(left)
        image_right = self.preprocess_input(right)
        input_dict = {'left_eye_image': image_left, 'right_eye_image': image_right,
                      'head_pose_angles': poses}
        start = time.time()
        # print('Input names', self.model.inputs)
        outputs = self.net.infer(input_dict)
        out = self.preprocess_outputs(outputs)

        # self.check_model(out)
        return out

    def check_model(self, coords):
        pass

    # print('Out shape', coords.shape)

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


class FaceDetect:
    """
    Class for the Face Detection Model.
    """

    def __init__(self, model_name, device, threshold=0.60):
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.threshold = threshold
        try:
            ie_core = IECore()
            self.model = ie_core.read_network(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Is it  the correct model path?")

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape
        self.net = None

    def load_model(self):
        """
        This method needs to be completed by you
        """
        ie_core = IECore()
        self.net = ie_core.load_network(self.model, self.device)

    def predict(self, image, initial_dimens):
        """
        This method needs to be completed by you
        """
        image_input = self.preprocess_input(image)
        input_dict = {self.input_name: image_input}
        start = time.time()
        outputs = self.net.infer(input_dict)
        out = self.preprocess_outputs(outputs)
        outs = self.draw_outputs(out, image, initial_dimens)
        return outs

    def check_model(self):
        pass

    def draw_outputs(self, coord, image, initial_dimens):
        """
        This method needs to be completed by you
        """
        outputs = []
        initial_w, initial_h = initial_dimens
        for obj in coord[0][0]:
            if obj[2] > self.threshold:
                xmin = int(obj[3] * initial_w)
                ymin = int(obj[4] * initial_h)
                xmax = int(obj[5] * initial_w)
                ymax = int(obj[6] * initial_h)
                outputs.append([xmin, ymin, xmax, ymax])
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)

        # img = image[outputs[0][1]:outputs[0][3], outputs[0][0]:outputs[0][2]]
        return outputs

    def preprocess_outputs(self, outputs):
        """
        This method needs to be completed by you
        """
        return outputs[self.output_name]

    def preprocess_input(self, image):
        """
        This method needs to be completed by you
        """
        n, c, h, w = self.input_shape
        if h > 0 and w > 0:
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
            image = np.moveaxis(image, -1, 0)
        return image



