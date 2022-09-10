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

os.environ["class_name"] = "Croissants, Dog"
os.environ["input_shape"] = "224"
os.environ["epochs"] = "10"
os.environ["batch_size"] = "10"

def main():

    class_names=Context.get_parameters("class_name")

    #read parameters from deployment config
    input_shape=int(Context.get_parameters("input_shape"))
    epochs=int(Context.get_parameters("epochs"))
    batch_size=int(Context.get_parameters("batch_size"))

    # load dataset
    #train_dataset_url = BaseConfig.train_dataset_url
    train_dataset_url="/home/lj1ang/Workspace/Python/NNFS/mindspore/datasets/DogCroissants/train"
    train_data = ImgDataset(data_type="train").parse(path=train_dataset_url,
                                    train=True,
                                    image_shape=input_shape,
                                    batch_size=batch_size)

    incremental_instance = IncrementalLearning(estimator=Estimator)
    return incremental_instance.train(train_data=train_data, epochs=epochs,
                                      batch_size=batch_size,
                                      class_names=class_names,
                                      input_shape=input_shape)

if __name__ == "__main__":
    main()

