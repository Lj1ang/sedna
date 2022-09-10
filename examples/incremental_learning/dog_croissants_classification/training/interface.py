# Copyright 2021 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division


import os

import PIL
import numpy as np
from PIL import Image
import mindspore as ms
import mindspore.nn as nn
from mindvision.engine.loss import CrossEntropySmooth
from mindvision.engine.callback import ValAccMonitor
from mobilenet_v2 import mobilenet_v2_fine_tune

os.environ['BACKEND_TYPE'] = 'MINDSPORE'

def preprocess(img:PIL.Image.Image):
    #image=Image.open(img_path).convert("RGB").resize((224 ,224))
    image=img.convert("RGB").resize((224,224))
    mean = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255])
    std = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255])
    image = np.array(image)
    image = (image - mean) / std
    image = image.astype(np.float32)

    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

class Estimator:


    def __init__(self,**kwargs):
        pass


    # TODO:save url
    # example : https://www.mindspore.cn/doc/programming_guide/zh-CN/r1.0/train.html#id3
    def train(self, train_data, valid_data=None, **kwargs):
        network=mobilenet_v2_fine_tune().get_model()
        network_opt=nn.Momentum(params=network.trainable_params(),learning_rate=0.01,momentum=0.9)
        network_loss=CrossEntropySmooth(sparse=True, reduction="mean", smooth_factor=0.1, classes_num=2)
        metrics = {"Accuracy" : nn.Accuracy}
        model=ms.Model(network, loss_fn=network_loss, optimizer=network_opt, metrics=metrics)
        num_epochs = 10
        model.train(num_epochs, train_data, callbacks=[ValAccMonitor(model, valid_data, num_epochs), ms.TimeMonitor])
        # save
        ms.save_checkpoint(network, "mobilenet_v2.ckpt")


    def evaluate(self,valid_data,model_path="",input_shape=(224,224),**kwargs):
        # load
        network = mobilenet_v2_fine_tune().get_model()
        ms.load_checkpoint("mobilenet_v2.ckpt", network)
        # eval
        network_loss = CrossEntropySmooth(sparse=True, reduction="mean", smooth_factor=0.1, classes_num=2)
        model = ms.Model(network, loss_fn=network_loss, optimizer=None, metrics={'acc'})
        acc=model.eval(valid_data, dataset_sink_mode=False)
        print(acc)
        return acc


    def predict(self, data, input_shape=None, **kwargs):
        # load
        network = mobilenet_v2_fine_tune().get_model()
        ms.load_checkpoint("mobilenet_v2.ckpt", network)
        model=ms.Model(network)
        # preprocess
        preprocessed_data=preprocess(data)
        # predict
        pre=model.predict(preprocessed_data)
        result=np.argmax(pre)
        class_name={0:"Croissants", 1:"Dog"}
        return class_name[result]

    def load(self, model_url):
        pass

    def save(self, model_path=None):
        pass










