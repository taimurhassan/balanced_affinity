from codebase.models.segnet import *
from tensorflow import keras
os.add_dll_directory(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0")
import tensorflow as tf

config = tf.compat.v1.ConfigProto(gpu_options = 
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)

# device_count = {'GPU': 1}
)

config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

model = resnet50_segnet(n_classes=7, input_height=576, input_width=768)

model.train(
    train_images =  r"C:\Users\fta71\PycharmProjects\pythonProject2\TrainingTensors2\train_images1",
    train_annotations =  r"C:\Users\fta71\PycharmProjects\pythonProject2\TrainingTensors2\train_annotations1",
	val_images =  r"C:\Users\fta71\PycharmProjects\pythonProject2\TrainingTensors2\val_images1",
    val_annotations =  r"C:\Users\fta71\PycharmProjects\pythonProject2\TrainingTensors2\val_annotations1",
    checkpoints_path = None , epochs=50, validate=True)

# model.summary()

# folder = r"C:\Users\fta71\PycharmProjects\pythonProject2\testingDataset\test_images"
# for filename in os.listdir(folder):
# 	out = model.predict_segmentation(inp=os.path.join(folder,filename),
# 	out_fname=os.path.join(r"C:\Users\fta71\PycharmProjects\pythonProject2\testdata\segmentation_results",filename))

print(model.evaluate_segmentation( inp_images_dir= r"C:\Users\fta71\PycharmProjects\pythonProject2\TestingTensors2\test_images1"  ,
	annotations_dir= r"C:\Users\fta71\PycharmProjects\pythonProject2\TestingTensors2\test_annotations"))