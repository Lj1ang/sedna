

## Prepare Model
auto-download

## Prepare for inference worker
```shell
mkdir -p /incremental_learning/infer/
mkdir -p /incremental_learning/he/
mkdir -p /data/dog_croissants/
mkdir /output
```

download dataset
```shell

???

```

download checkpoint
```shell
mkdir -p /model/base_model
mkdir -p /model/deploy_model
cd /model/base_model
wget https://download.mindspore.cn/vision/classification/mobilenet_v2_1.0_224.ckpt
```
## build docker file
```shell
$  docker build -f incremental-learning-dog-croissants-classification.Dockerfile -t test/dog:v0.1 .

```

## Create Incremental Job
```shell
WORKER_NODE="edge-node" 
```
Create Dataset
```shell
kubectl create -f - <<EOF
apiVersion: sedna.io/v1alpha1
kind: Dataset
metadata:
  name: incremental-dataset
spec:
  url: "/data/dog_croissants/train_data/train_data.txt"
  format: "txt"
  nodeName: $WORKER_NODE
EOF
```
Create initial Model to simulate the inital model in incremental learning scenoario
```shell
kubectl create -f - <<EOF
apiVersion: sedna.io/v1alpha1
kind: Model
metadata:
  name: initial-model
spec:
  url : "/models/base_model/mobilenet_v2_1.0_224.ckpt"
  format: "ckpt"
EOF
```
Create Deploy Model
```shell
kubectl create -f - <<EOF
apiVersion: sedna.io/v1alpha1
kind: Model
metadata:
  name: deploy-model
spec:
  url : "/models/deploy_model/saved_model.ckpt"
  format: "ckpt"
EOF
```
create the job
```shell
IMAGE=
kubectl create -f - <<EOF
apiVersion: sedna.io/v1alpha1
kind: IncrementalLearningJob
metadata:
  name: dog-croissants-classification-demo
spec:
  initialModel:
    name: "initial-model"
  dataset:
    name: "incremental-dataset"
    trainProb: 0.9
  trainSpec:
    template:
      spec:
        nodeName: $WORKER_NODE
        containers:
          - image: $IMAGE
            name: train-worker
            imagePullPolicy: IfNotPresent
            args: [ "train.py" ]
            env:
              - name: "batch_size"
                value: "8"
              - name: "epochs"
                value: "10"
              - name: "input_shape"
                value: "224"
              - name: "class_names"
                value: "Croissants, Dog"
    trigger:
      checkPeriodSeconds: 60
      timer:
        start: 01:00
        end: 10:00
      condition:
        operator: ">"
        threshold: 20
        metric: num_of_samples
  evalSpec:
    template:
      spec:
        nodeName: $WORKER_NODE
        containers:
          - image: $IMAGE
            name: eval-worker
            imagePullPolicy: IfNotPresent
            args: [ "eval.py" ]
            env:
              - name: "input_shape"
                value: "224"
              - name: "class_names"
                value: "Croissants, Dog"
  deploySpec:
    model:
      name: "deploy-model"
      hotUpdateEnabled: true
      pollPeriodSeconds: 60
    trigger:
      condition:
        operator: ">"
        threshold: 0.1
        metric: precision_delta
    hardExampleMining:
      name: "Random"
      parameters:
        - key: "random_ratio"
          value: "0.3"
    template:
      spec:
        nodeName: $WORKER_NODE
        containers:
          - image: $IMAGE
            name: infer-worker
            imagePullPolicy: IfNotPresent
            args: [ "inference.py" ]
            env:
              - name: "input_shape"
                value: "224"
              - name: "infer_url"
                value: "/incremental_learning/infer"
              - name: "HE_SAVED_URL"
                value: "/incremental_learning/he"
            volumeMounts:
              - name: localinferdir
                mountPath: /incremental_learning/infer
              - name: hedir
                mountPath: /incremental_learning/he
            resources: # user defined resources
              limits:
                memory: 2Gi
        volumes: # user defined volumes
          - name: localinferdir
            hostPath:
              path: /incremental_learning/infer
              type: DirectoryOrCreate
          - name: hedir
            hostPath:
              path: /incremental_learning/he
              type: DirectoryOrCreate
  outputDir: "/output"
EOF
```
## trigger
```shell
cd /data/helmet_detection
wget  https://kubeedge.obs.cn-north-1.myhuaweicloud.com/examples/helmet-detection/dataset.tar.gz
tar -zxvf dataset.tar.gz
```