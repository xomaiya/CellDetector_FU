from pathlib import Path
from SlideRunner.dataAccess.database import Database
from get_slides import get_slides
import numpy as np

import os
import tensorflow as tf
import time
from label_encoder import LabelEncoder
from retinaNet import get_backbone, RetinaNet
from losses import RetinaNetLoss
from preprocessing import preprocess_data
from dataset import MyDataset

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))

size=256
path = Path('../MITOS_WSI_CMC/')

database = Database()
database.open(str(path/'./databases/MITOS_WSI_CMC_MEL.sqlite'))

slidelist_test_1 = ['14','18','3','22','10','15','21']
slidelist_test_2 = ['1','20','17','5','2','11','16']
slidelist_test_3 = ['13','7','19','8','6','9', '12']
# slidelist_test = slidelist_test_1
slidelist_test = []

lbl_bbox, files = get_slides(slidelist_test=slidelist_test,
                                   negative_class=1,
                                   size=size,
                                   database=database,
                                   basepath='/scratch/olesia/WSI')

img2bbox = dict(zip(files, np.array(lbl_bbox)))
get_y_func = lambda o:img2bbox[o]

train_data = MyDataset(files[:17], lbl_bbox[:17])
val_data = MyDataset(files[17:], lbl_bbox[17:])

train_dataset = train_data.inf_dataset
val_dataset = val_data.dataset

model_dir = "retinanet_tumor/"
label_encoder = LabelEncoder()

num_classes = 4
batch_size = 16

learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
learning_rate_boundaries = [125, 250, 500, 240000, 360000]
learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    boundaries=learning_rate_boundaries, values=learning_rates
)

resnet50_backbone = get_backbone()
loss_fn = RetinaNetLoss(num_classes)
model = RetinaNet(num_classes, resnet50_backbone)

optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
model.compile(loss=loss_fn, optimizer=optimizer)

callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
        monitor="loss",
        save_best_only=False,
        save_weights_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.TensorBoard(
        log_dir=f'logs/{time.time()}',
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        update_freq='batch',
        profile_batch=2,
        embeddings_freq=0,
        embeddings_metadata=None
    )
]

autotune = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=autotune)
train_dataset = train_dataset.shuffle(8 * batch_size)
train_dataset = train_dataset.padded_batch(
    batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True
)
train_dataset = train_dataset.map(
    label_encoder.encode_batch, num_parallel_calls=autotune
)
train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
train_dataset = train_dataset.prefetch(autotune)

val_dataset = val_dataset.map(preprocess_data, num_parallel_calls=autotune)
val_dataset = val_dataset.padded_batch(
    batch_size=1, padding_values=(0.0, 1e-8, -1), drop_remainder=True
)
val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
val_dataset = val_dataset.prefetch(autotune)

# train_dataset = train_dataset.take(10)

train_steps_per_epoch = 1000
val_steps_per_epoch = 10

train_steps = 4 * 100000
epochs = train_steps // train_steps_per_epoch

# epochs = 1

# Running 100 training and 50 validation steps,
# remove `.take` when training on the full dataset

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=callbacks_list,
    verbose=1,
)

