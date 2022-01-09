import os
import tqdm
import tensorflow.keras
import tensorflow
from tensorflow.keras import layers
from tensorflow.keras import models
import sklearn
from tensorflow.keras import optimizers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.resnet import ResNet152
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.xception import Xception
from affinity_loss import *
from affinity import *
from losses4 import ArcLoss
from focal_loss import categorical_focal_loss, binary_focal_loss
from losses3 import triplet_loss_all
# from pytorch_metric_learning import losses
import tensorflow_addons as tfa
from loss_functions import CircleLoss
from ride import tversky, Combo_loss, lovasz, marginal_loss, range_loss, AM_logits
from Balanced_Affinity_loss import *

original_dataset_dir = r'C:\Users\fta71\PycharmProjects\pythonProject2\SIXray10'


train_dir =os.path.join(original_dataset_dir+"\\Training")
test_dir =os.path.join(original_dataset_dir+"\\Testing")
# val_dir =os.path.join(original_dataset_dir+"\\Validation")


# loss = balanced_affinity_loss(0.1)
loss= affinity_loss(0.1)
# loss = categorical_focal_loss(0.25, gamma=2.)
#loss = tf.keras.losses.BinaryCrossentropy(name="binary_crossentropy")

# conv_base =DenseNet201(input_shape = (120, 120, 3), # Shape of our images
#                   include_top = False, # Leave out the last fully connected layer
#                   weights = 'imagenet')

conv_base = ResNet50(input_shape=(120, 120, 3),  # Shape of our images
                     include_top=False,  # Leave out the last fully connected layer
                     weights='imagenet')

# conv_base = VGG19(input_shape = (120, 120, 3), # Shape of our images
#                   include_top = False, # Leave out the last fully connected layer
#                   weights = 'imagenet')

# conv_base = VGG16(input_shape=(120, 120, 3),  # Shape of our images
#                   include_top=False,  # Leave out the last fully connected layer
#                   weights='imagenet')

# conv_base = InceptionV3(input_shape=(120, 120, 3),  # Shape of our images
#                   include_top=False,  # Leave out the last fully connected layer
#                   weights='imagenet')

# conv_base = InceptionResNetV2(input_shape=(120, 120, 3),  # Shape of our images
#                   include_top=False,  # Leave out the last fully connected layer
#                   weights='imagenet')

# conv_base = Xception(input_shape=(120, 120, 3),  # Shape of our images
#                   include_top=False,  # Leave out the last fully connected layer
#                   weights='imagenet')

# conv_base =MobileNet(input_shape = (120, 120, 3), # Shape of our images
#                   include_top = False, # Leave out the last fully connected layer
#                   weights = 'imagenet')

conv_base.trainable = False

# conv_base.trainable = True
# set_trainable = False
# for layer in conv_base.layers:
#     if layer.name == 'block5_conv1' and layer.name == 'block5_conv2' and layer.name == 'block5_conv3'  and layer.name == 'block4_conv1' and layer.name == 'block4_conv2' and layer.name == 'block4_conv3':
#         set_trainable = True
#     if set_trainable:
#         layer.trainable = True
#     else:
#         layer.trainable = False

model = models.Sequential()

model.add(conv_base)
model.add(layers.GlobalAvgPool2D())
# model.add(layers.BatchNormalization())
# model.add(layers.Flatten())
# model.add(layers.Dense(16, activation='relu'))
# model.add(layers.Dropout(0.1))
# model.add(layers.Dense(2, activation='softmax'))
model.add(ClusteringAffinity(2, 1, 130))
# model.summary()
#'categorical_crossentropy'
opt = optimizers.Adam(learning_rate=0.001)
model.compile(loss=loss,
              optimizer=opt, metrics=["acc"], run_eagerly= True)

train_datagen = ImageDataGenerator(rescale=1./255)

# val_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(120, 120), batch_size= 32,
                                                        class_mode='categorical')

# Val_generator = val_datagen.flow_from_directory(val_dir, target_size=(80, 80), batch_size= 2,
#                                                             class_mode='categorical')

Test_generator = test_datagen.flow_from_directory(test_dir, target_size=(120, 120), batch_size= 32,
                                                           class_mode='categorical')

history = model.fit_generator(train_generator, steps_per_epoch=40, epochs=10)

# (loss, tf.compat.v1.metrics.average_precision_at_k) = model.evaluate(Test_generator)
print(model.evaluate(x= Test_generator, steps= 80))

# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(acc) + 1)
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()

# print(model.evaluate(Test_generator, steps= 2))

# Y_pred = model.predict_generator(Test_generator)
# y_pred = np.argmax(Y_pred ,axis =1)
# pre = average_precision_score(Test_generator.classes, Y_pred, average='macro', sample_weight=None)
# F1 = f1_score(Test_generator.classes, Y_pred, average='macro', sample_weight=None)
# call = recall_score(Test_generator.classes, Y_pred, average='macro', sample_weight=None)
# m = Test_generator.classes
# acc = accuracy_score(m, Y_pred)
# print(pre)
# print(F1)
# print(call)
# print(acc)