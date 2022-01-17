import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential, layers, optimizers
from tensorflow.keras.applications import EfficientNetB5 as Net
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet import ResNet152
from tensorflow.keras.applications.densenet import DenseNet121

# Hyper parameters
batch_size = 64

# B0的輸入是 224*224*3
width = 456
height = 456
epochs = 100
NUM_TRAIN = 3793
NUM_TEST = 3793
dropout_rate = 0.02
input_shape = (height, width, 3)
data_dir = "./images"

# train data
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(height, width),
  batch_size=batch_size)
class_names = train_ds.class_names
print(class_names)


# Build Model
conv_base = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
model = Sequential()
model.add(conv_base)
model.add(layers.GlobalAveragePooling2D(name="gap"))
model.add(layers.BatchNormalization())

if dropout_rate > 0:
    model.add(layers.Dropout(dropout_rate, name="dropout_out"))
model.add(layers.Dense(9, activation='softmax', name="fc_out"))

# 模型蓋覽
model.summary()

# 凍結卷積層，不參與訓練
conv_base.trainable = False

model.compile(
        loss='categorical_crossentropy', 
        optimizer = optimizers.SGD(lr=0.01),
        # optimizer=optimizers.RMSprop(lr=2e-5),
        metrics=['acc'])

import os
import glob
test_dir = "時代不詳"
txt = []
images = [_ for _ in os.listdir(test_dir) if _.endswith(".jpg")]
for image in images:
    img = tf.keras.utils.load_img(
            test_dir + "/" + image, target_size=(height, width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    
    result = image + "  |預測:" + class_names[np.argmax(score)] + "  可信度:" + str(100 * np.max(score))
    
    txt.append(image)
    txt.append("預測:【" + class_names[np.argmax(score)] + "】")
    txt.append("可信度:【" + str(round(100 * np.max(score), 2)) + "%】")
    txt.append("\n")
    
with open('./時代不詳/predict.txt', 'w') as f:
    for i in txt:
        f.write(i)
        f.write('\n')


