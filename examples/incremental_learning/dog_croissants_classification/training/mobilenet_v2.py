import mindspore as ms
from mindvision.classification.models import mobilenet_v2
from mindvision.dataset import DownLoad



class mobilenet_v2_fine_tune:
    # TODO: save model
    def __init__(self):
        models_url = "https://download.mindspore.cn/vision/classification/mobilenet_v2_1.0_224.ckpt"
        dl=DownLoad()
        dl.download_url(models_url)
        self.network = mobilenet_v2(num_classes=2, resize=224)
        self.param_dict = ms.load_checkpoint("./mobilenet_v2_1.0_224.ckpt")
        self.filter_list=[x.name for x in self.network.head.classifier.get_parameters()]

    def filter_ckpt_parameter(origin_dict, param_filter):
        for key in list(origin_dict.keys()):
            for name in param_filter:
                if name in key:
                    print("Delete parameter from checkpoint: ", key)
                    del origin_dict[key]
                    break

    def get_model(self):
        self.filter_ckpt_parameter(self.param_dict, self.filter_list)
        ms.load_param_into_net(self.network, self.param_dict)
        return self.network