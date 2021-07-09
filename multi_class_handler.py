import base64
from utils import *
from ts.utils.util import map_class_to_label
from ts.torch_handler.vision_handler import VisionHandler
from ts.torch_handler.image_classifier import ImageClassifier

class MultiImageClassifier(ImageClassifier):
    def lala(l): return l
    # # def __init__(self, model):
    #     # super().__init__()

    # """
    # ImageClassifier handler class. This handler takes an image
    # and returns the name of object in that image.
    # """

    # # topk = 5
    # # These are the standard Imagenet dimensions
    # # and statistics
    # image_size = 512
    # mean, std = tensor([0.4850, 0.4560, 0.4060]), tensor([0.2290, 0.2240, 0.2250])
    # image_processing = instant_tfms(h=image_size, w=image_size, img_mean=mean, img_std=std, test_tfms=False)
    # # image_processing = transforms.Compose([
    # #     transforms.Resize(600),
    # #     transforms.CenterCrop(image_size),
    # #     transforms.ToTensorV2(),
    # #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    # #                          std=[0.229, 0.224, 0.225])
    # # ])

    # def preprocess(self, data):
    #     """The preprocess function of MNIST program converts the input data to a float tensor

    #     Args:
    #         data (List): Input data from the request is in the form of a Tensor

    #     Returns:
    #         list : The preprocess function returns the input image as a list of float tensors.
    #     """
    #     images = []

    #     for row in data:
    #         # Compat layer: normally the envelope should just return the data
    #         # directly, but older versions of Torchserve didn't have envelope.
    #         image = row.get("data") or row.get("body")
    #         if isinstance(image, str):
    #             # if the image is a string of bytesarray.
    #             image = base64.b64decode(image)

    #         # If the image is sent as bytesarray
    #         if isinstance(image, (bytearray, bytes)):
    #             image = Image.open(io.BytesIO(image))
    #             image = apply_tfms(image, self.image_processing)
    #             print(f'IMAAGGEEE SSSHHHAAAPPPEEEE: {image.shape}')
    #         else:
    #             # if the image is a list
    #             image = torch.FloatTensor(image)

    #         images.append(image)

    #     return torch.stack(images).to(self.device)

    # def set_max_result_classes(self, topk):
    #     self.topk = topk

    # def get_max_result_classes(self):
    #     return self.topk

    # def postprocess(self, data):

    #     thresh = 0.65
    #     p_idx = []
    #     idx = F.sigmoid(data) > thresh
    #     idx = idx.tolist()
    #     classes = [[j for j,x in enumerate(id) if x] for id in idx]
    #     probs = [data[i][id].tolist() for i,id in enumerate(classes)]
    #     return map_class_to_label(probs, self.mapping, classes)
