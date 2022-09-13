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
import os

from sedna.common.config import Context, BaseConfig
from sedna.core.incremental_learning import IncrementalLearning

from interface import Estimator
from dataset import ImgDataset

def main():

    class_names=Context.get_parameters("class_name")
    print(Context.get_parameters("model_path"))
    #read parameters from deployment config
    input_shape=int(Context.get_parameters("input_shape"))
    batch_size=int(Context.get_parameters("batch_size"))

    # load dataset
    #train_dataset_url = BaseConfig.train_dataset_url
    valid_dataset_url="/home/lj1ang/Workspace/Python/NNFS/mindspore/datasets/DogCroissants/val"
    valid_data=ImgDataset(data_type="eval").parse(path=valid_dataset_url,
                                                  train=False,
                                                  image_shape=input_shape,
                                                  batch_size=batch_size)
    incremental_instance = IncrementalLearning(estimator=Estimator)
    return incremental_instance.evaluate(valid_data,
                                      class_names=class_names,
                                      input_shape=input_shape)

if __name__ == "__main__":
    main()

