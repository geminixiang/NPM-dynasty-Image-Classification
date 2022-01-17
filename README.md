# Quick Start
## miniconda
1. Install requirements
```bash
conda create -n tf2 python=3.7
conda activate tf2
pip install -r requirements.txt
```

2. Crawler
```bash
jupyter lab --allow-root
```
and using ipython file to download images from [NPM](https://theme.npm.edu.tw/opendata/DigitImageSets.aspx?Key=^22^11&pageNo=1)

3. Traing model
```bash
python train.py
```
It'll output `my_model.h5` to models folder.

4. Predict
```bash
python predict.py
```
It'll output `predict.txt` to `時代不詳` folder.

## Using different [applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications)
```
# VGG19
4:00-6:00
Epoch 60/100
59/59 [==============================] - 90s 1s/step - loss: 1.1011 - acc: 0.5554 - val_loss: 1.0712 - val_acc: 0.5744
Epoch 61/100
59/59 [==============================] - 90s 1s/step - loss: 1.0988 - acc: 0.5605 - val_loss: 1.0791 - val_acc: 0.5662
Epoch 62/100
59/59 [==============================] - 90s 1s/step - loss: 1.1012 - acc: 0.5619 - val_loss: 1.0762 - val_acc: 0.5715

# Resnet152
05:56-9:00
Epoch 98/100
59/59 [==============================] - 157s 3s/step - loss: 1.2147 - acc: 0.5070 - val_loss: 1.2058 - val_acc: 0.5188
Epoch 99/100
59/59 [==============================] - 156s 3s/step - loss: 1.2207 - acc: 0.5121 - val_loss: 1.2048 - val_acc: 0.5164
Epoch 100/100
59/59 [==============================] - 157s 3s/step - loss: 1.2155 - acc: 0.5056 - val_loss: 1.2123 - val_acc: 0.5098

# DenseNet121
Epoch 67/100
59/59 [==============================] - 70s 1s/step - loss: 0.8987 - acc: 0.6483 - val_loss: 0.8719 - val_acc: 0.6584
Epoch 68/100
59/59 [==============================] - 70s 1s/step - loss: 0.9042 - acc: 0.6457 - val_loss: 0.8497 - val_acc: 0.6743
Epoch 69/100
59/59 [==============================] - 70s 1s/step - loss: 0.9005 - acc: 0.6460 - val_loss: 0.8584 - val_acc: 0.6668
```
