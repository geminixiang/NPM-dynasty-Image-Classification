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

train_dir = './images'
validation_dir = './images'

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

validation_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(height, width),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(height, width),
        batch_size=batch_size,
        class_mode='categorical')


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

history_tl = model.fit_generator(
      train_generator,
      steps_per_epoch= NUM_TRAIN // batch_size,
      epochs=epochs,
      validation_data=validation_generator,
      validation_steps= NUM_TEST // batch_size,
      verbose=1,
      use_multiprocessing=True,
      workers=4
)


# 模型文件儲存
model.save('./models/my_model.h5')




def plot_training(history):

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_x = range(len(acc))

    plt.plot(epochs_x, acc, 'bo', label='Training acc')
    plt.plot(epochs_x, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs_x, loss, 'bo', label='Training loss')
    plt.plot(epochs_x, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


