# Parking-Spot-Detection

This repository contains the code that was used to train a parking spot detection model. While the dataset is not available, you can download the trained models and test them with the server. There is also a demo mobile app ([source](https://gitlab.com/avshon1/mlparking)).

# Download Models

The models can be downloaded from [Google Drive](https://drive.google.com/file/d/1wUYoObciki2CjEBCQ5eKP95q40X78fKZ/view?usp=sharing). The zip file contains two folders for ResNet-18 and MobileNetV3. Each folder contains a saved torchscript model and a model description file.

# Test Models

To test the model, you first need to install the required dependencies:

```sh
pip install -r requirements/requirements.txt
```

One of the downloaded models and its description should be moved to `/api`:

```sh
mv /path/to/model.pt api/model.pt
mv /path/to/model_desc.json api/model_desc.json
```

After that you can launch the local test server:
```sh
uvicorn api.app:app
```

To test that server is working, you can open `localhost:8000` in browser. To use the server you can go to `http://localhost:8000/docs` -> `predict` -> `Try it out` and choose one of the test images in `/api`. You can also use this command:

```sh
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image_file=@api/test_free_spots.jpg;type=image/jpeg'
```

# Dataset and Model Descriptions

The models were trained on a dataset containing 5000 images (TRAIN/VAL/TEST - 4275/225/500) of roads and parking spaces from Moscow. The inference times were tested on IPhone XS Max.

## MobileNetV3

```sh
python training/run_experiment.py --model_class MobileNetV3 --mobilenetv3_size=small --use_torchvision_model --data_class ParkingSpots --optimizer AdamW --lr 0.00035 --weight_decay 0.25 --gpus 1 --precision 16 --use_local --max_epochs 16 --batch_size 64 --seed 0 --num_workers 8 --color_jitter 0.3 --horizontal_flip --degrees_affine 30 --translate_affine 0.25 --scale_margin_affine 0.3 --shear_affine 5 --lr_scheduler OneCycleLR --random_erasing  --es_patience 30 --save_torchscript
```


|      |Accuracy| Loss |
|------|--------|------|
|Train | 0.7722 |0.501 |
|Val   | 0.6987 |0.7399|
|Test  | 0.6706 |0.8363|

| Total Avg. Time | Avg. Inference Time Only |
|--------|------|
| 0.482 |0.06 |


## ResNet-18

```sh
python training/run_experiment.py --model_class ResNet --resnet_type resnet18 --use_torchvision_model --data_class ParkingSpots --optimizer AdamW --lr 0.00035 --weight_decay 0.25 --gpus 1 --precision 16 --use_local --max_epochs 15 --batch_size 42 --seed 0 --num_workers 4 --color_jitter 0.3 --horizontal_flip --degrees_affine 30 --translate_affine 0.25 --scale_margin_affine 0.4 --shear_affine 5 --lr_scheduler OneCycleLR --random_erasing --use_lr_monitor --es_patience 30 --save_torchscript
```

|      |Accuracy| Loss |
|------| ------ | ---- |
|Train | 0.7966 |0.3914|
|Val   | 0.7399 |0.6817|
|Test  | 0.7022 |0.7719|

| Total Avg. Time | Avg. Inference Time Only |
|--------|------|
| 1.224 |0.8 |

# References

- Test Images from Wikimedia Commons:
  - [test_free_spots.jpg](https://commons.wikimedia.org/wiki/File:20200925-parking-lined-perpendicular-large.jpg)
  - [test_taken_spots.jpg](https://commons.wikimedia.org/wiki/File:20200925-parking-unlined-perpendicular.jpg)
  - [test_free_error.jpg](https://commons.wikimedia.org/wiki/File:Hameau_des_Renaudi%C3%A8res,_Carquefou_-_01.jpg)
