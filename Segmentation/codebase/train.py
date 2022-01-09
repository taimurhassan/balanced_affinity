import argparse
import json
from .data_utils.data_loader import image_segmentation_generator, \
    verify_segmentation_dataset, ImageSegmentationGen
import os
import glob
import six
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
import tensorflow.keras
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from affinity_loss import *
from Balanced_Affinity_loss import *
from losses4 import ArcLoss
from focal_loss import categorical_focal_loss
from losses3 import triplet_loss_all
# from pytorch_metric_learning import losses
import tensorflow_addons as tfa
from loss_functions import CircleLoss
from ride import tversky, Combo_loss, lovasz, marginal_loss, range_loss, AM_logits
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler, Callback

def loss1(y_true, y_pred):
    def dice_loss(y_true, y_pred):
        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
        denominator = tf.reduce_sum(y_true + y_pred, axis=(1,2,3))

        return tf.reshape(1 - numerator / denominator, (-1, 1, 1))

    return dice_loss(y_true, y_pred)
    
def customLoss(yTrue,yPred):
    dice = K.mean(1-(2 * K.sum(yTrue * yPred))/(K.sum(yTrue + yPred)))
    return K.mean(dice + tensorflow.keras.losses.categorical_crossentropy(yTrue,yPred))
	
    
def find_latest_checkpoint(checkpoints_path, fail_safe=True):

    def get_epoch_number_from_path(path):
        return path.replace(checkpoints_path, "").strip(".")

    # Get all matching files
    all_checkpoint_files = glob.glob(checkpoints_path + ".*")
    # Filter out entries where the epoc_number part is pure number
    all_checkpoint_files = list(filter(lambda f: get_epoch_number_from_path(f).isdigit(), all_checkpoint_files))
    if not len(all_checkpoint_files):
        # The glob list is empty, don't have a checkpoints_path
        if not fail_safe:
            raise ValueError("Checkpoint path {0} invalid".format(checkpoints_path))
        else:
            return None

    # Find the checkpoint file with the maximum epoch
    latest_epoch_checkpoint = max(all_checkpoint_files, key=lambda f: int(get_epoch_number_from_path(f)))
    return latest_epoch_checkpoint

#loss= categorical_focal_loss(0.25, gamma=2.)
#loss = triplet_loss_all()
#loss = CircleLoss(gamma = 64, margin = 0.25, batch_size = None, reduction='auto', name=None)
#loss = affinity_loss(0.64)
#loss = tfa.losses.GIoULoss( mode = 'giou', reduction= tf.keras.losses.Reduction.AUTO, name= 'giou_loss')
#loss = tversky()
#loss = Combo_loss()
#loss = lovasz()
#loss = marginal_loss()
#loss = Ran()
#loss = AM_logits()
loss = balanced_affinity_loss(0.1)

def train(model,
          train_images,
          train_annotations,
          input_height=None,
          input_width=None,
          n_classes=None,
          verify_dataset=False,
          checkpoints_path=None,
          epochs=20,
          batch_size=16,
          validate=False,
          val_images=None,
          val_annotations=None,
          val_batch_size=20,
          auto_resume_checkpoint=False,
          load_weights=None,
          steps_per_epoch=20,
          optimizer_name='Adam',
		  do_augment=False, 
		  classifier=None
          ):

    #from .models.all_models import model_from_name
    # check if user gives model name instead of the model object
    #if isinstance(model, six.string_types):
        # create the model from the name
    #    assert (n_classes is not None), "Please provide the n_classes"
    #    if (input_height is not None) and (input_width is not None):
    #        model = model_from_name[model](
    #            n_classes, input_height=input_height, input_width=input_width)
    #    else:
    #        model = model_from_name[model](n_classes)

    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width

    if validate:
        assert val_images is not None
        assert val_annotations is not None
#loss="categorical_crossentropy",#
    if optimizer_name is not None:
        model.compile(loss = loss, #loss=lambda yTrue, yPred: customLoss(yTrue, yPred),
                      optimizer=optimizer_name,
                      metrics=['accuracy'],  run_eagerly= True)

    if checkpoints_path is not None:
        with open(checkpoints_path+"_config.json", "w") as f:
            json.dump({
                "model_class": model.model_name,
                "n_classes": n_classes,
                "input_height": input_height,
                "input_width": input_width,
                "output_height": output_height,
                "output_width": output_width
            }, f)

    if load_weights is not None and len(load_weights) > 0:
        print("Loading weights from ", load_weights)
        model.load_weights(load_weights)

    if auto_resume_checkpoint and (checkpoints_path is not None):
        latest_checkpoint = find_latest_checkpoint(checkpoints_path)
        if latest_checkpoint is not None:
            print("Loading the weights from latest checkpoint ",
                  latest_checkpoint)
            model.load_weights(latest_checkpoint)

    if verify_dataset:
        print("Verifying training dataset")
        verified = verify_segmentation_dataset(train_images, train_annotations, n_classes)
        assert verified
        if validate:
            print("Verifying validation dataset")
            verified = verify_segmentation_dataset(val_images, val_annotations, n_classes)
            assert verified

    train_gen = ImageSegmentationGen(
        train_images, train_annotations,  batch_size,  n_classes,
        input_height, input_width, output_height, output_width , do_augment=do_augment )

    if validate:
        val_gen = ImageSegmentationGen(
            val_images, val_annotations,  val_batch_size,
            n_classes, input_height, input_width, output_height, output_width)

    if not validate:
        for ep in range(epochs):
            print("Starting Epoch ", ep)
            history = model.fit_generator(train_gen, steps_per_epoch,  epochs=1, workers=16)
            if checkpoints_path is not None:
                model.save_weights(checkpoints_path + "." + str(ep))
                print("saved ", checkpoints_path + ".model." + str(ep))
            
    else:
        for ep in range(epochs):
            print("Starting Epoch ", ep)
            history = model.fit_generator(train_gen, steps_per_epoch, validation_data=val_gen, validation_steps=5,  epochs=1, workers=16)

            if checkpoints_path is not None:
                model.save_weights(checkpoints_path + "." + str(ep))
                print("saved ", checkpoints_path + ".model." + str(ep))
            print("Finished Epoch", ep)




