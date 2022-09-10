from sedna.core.joint_inference import JointInference, BigModelService
from sedna.core.federated_learning import FederatedLearning
from sedna.core.incremental_learning import IncrementalLearning
from sedna.core.lifelong_learning import LifelongLearning

import os

# Keras
import keras
import tensorflow as tf
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Dropout
from keras.models import Sequential

os.environ['BACKEND_TYPE'] = 'KERAS'


def KerasEstimator():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation="relu", strides=(2, 2),
                     input_shape=(128, 128, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation="softmax"))

    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)
    metrics = [keras.metrics.categorical_accuracy]
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model

# get hard exmaple mining algorithm from config
hard_example_mining = IncrementalLearning.get_hem_algorithm_from_config(
    threshold_img=0.9
)

Estimator=KerasEstimator()
# create Incremental Learning infernece instance
il_job = IncrementalLearning(
    estimator=Estimator,
    hard_example_mining=hard_example_mining
)

