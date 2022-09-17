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
    # base_model_url means the low accuracy model
    base_model_url=Context.get_parameters("base_model_url")
    # model_url means the checkpoint file that has been trained
    deploy_model_url = Context.get_parameters("deploy_model_url")
    #read parameters from deployment config
    input_shape=int(Context.get_parameters("input_shape"))
    epochs=int(Context.get_parameters('epochs'))
    batch_size=int(Context.get_parameters("batch_size"))

    # load dataset
    train_dataset_url=Context.get_parameters("TRAIN_DATASET_URL")
    valid_dataset_url=Context.get_parameters("TEST_DATASET_URL")
    train_data = ImgDataset(data_type="train").parse(path=train_dataset_url,
                                                     train=True,
                                                     image_shape=input_shape,
                                                     batch_size=batch_size)
    valid_data=ImgDataset(data_type="eval").parse(path=valid_dataset_url,
                                                  train=False,
                                                  image_shape=input_shape,
                                                  batch_size=batch_size)
    incremental_instance = IncrementalLearning(estimator=Estimator)
    return incremental_instance.train(train_data=train_data,
                                      base_model_url=base_model_url,
                                      deploy_model_url=deploy_model_url,
                                      valid_data=valid_data,
                                      epochs=epochs)

if __name__ == "__main__":
    main()

