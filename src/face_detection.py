import numpy as np
import time
import cv2
from openvino.inference_engine import IENetwork, IECore


class FaceDetection:
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

        start = time.time()
        ie_core = IECore()
        self.net = ie_core.load_network(self.model, self.device, num_requests=4)
        print('[1] Loaded in  {:.3f} s'.format(time.time()-start))

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
        # self.performance_counter(0)
        return outs



    def performance_counter(self, request_id):
        """
        Queries performance measures per layer to get feedback of what is the
        most time consuming layer.
        :param request_id: Index of Infer request value. Limited to device capabilities
        :return: Performance of the layer
        """

        # perf_count = self.model.requests[request_id].get_perf_counts()
        # print('Performance count :')
        # print(self.net.requests[0].get_perf_counts())
        perf_count = self.net.requests[request_id].get_perf_counts()
        return perf_count

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
                # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1)

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
