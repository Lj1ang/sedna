

## Prepare Model
auto-download

## Prepare for inference worker
```shell
mkdir -p /incremental_learning/data/
mkdir -p /incremental_learning/he/
mkdir -p /data/helmet_detection ? 
mkdir /output
```

download dataset
```shell
cd /incremental_learning/data/
wget https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/beginner/DogCroissants.zip
unzip DogCroissants.zip
```

download checkpoint
```shell
mkdir -p /model/base_model
cd /model/base_model
wget "https://download.mindspore.cn/vision/classification/mobilenet_v2_1.0_224.ckpt"
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
  url: "/incremental_ learning/data/DogCroissants/train"
  format: "image"
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
    trainProb: 0.8
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
                value: "10"
              - name: "epochs"
                value: "10"
              - name: "input_shape"
                value: "224"
              - name: "class_names"
                value: "Croissants, Dog"
    trigger:
      checkPeriodSeconds: 60
      timer:
        start: 02:00
        end: 20:00
      condition:
        operator: ">"
        threshold: 10
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
      hotUpdateEnabled: false
      pollPeriodSeconds: 60
    trigger:
      condition:
        operator: ">"
        threshold: 0.1
        metric: precision_delta
    hardExampleMining:
      name: "IBT"
      parameters:
        - key: "threshold_img"
          value: "0.95"
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
                value: "/home/data/DogCroissants/infer/croissants.jpg"
              - name: "HE_SAVED_URL"
                value: "/he_saved_url"
            volumeMounts:
              - name: localinferdir
                mountPath: /home/data/DogCroissants/infer
              - name: hedir
                mountPath: /he_saved_url
            resources: # user defined resources
              limits:
                memory: 2Gi
        volumes: # user defined volumes
          - name: localinferdir
            hostPath:
              path: /incremental_learning/data/DogCroissants/infer
              type: DirectoryOrCreate
          - name: hedir
            hostPath:
              path: /incremental_learning/he/
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