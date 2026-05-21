#!/usr/bin/env python3
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd

# --------------------------
# Load or define all variables used in this cell
# Load the EuroSAT dataset (version 2.0.0)
# We load all splits in a single tfds.load() call:
#   - train_ds: first 80% of the original training split
#   - val_ds: next 10% of the original training split
#   - test_ds: last 10% of the original training split

import seaborn as sns
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow import keras
# your code
(ds_splits, info) = tfds.load(
    "eurosat:2.0.0",                         # dataset name with explicit version
    split=[
        "train[:80%]",                       # first 80% of the original training split
        "train[80%:90%]",                   # next 10%  -> validation
        "train[90%:]",                      # last 10%  -> test
    ],
    with_info=True,                         # also return DatasetInfo object
)

# Unpack the three datasets
train_ds, val_ds, test_ds = ds_splits


# Extract image shape and label shape from the dataset metadata
image_shape = info.features["image"].shape
label_shape = info.features["label"].shape

# Extract number of classes and class names
num_classes = info.features["label"].num_classes
class_names = info.features["label"].names

# Extract number of training examples from the original split
num_train_examples = info.splits["train"].num_examples

# your code
def count_occurences(ds, info):
    # 1) Get number of classes from the DatasetInfo
    num_classes = info.features["label"].num_classes

    # 2) Initialize counts to zero for each class
    num_occurences = np.zeros(num_classes, dtype=np.int32)
    for item in ds:
        # item["label"] is a scalar tf.Tensor
        label = int(item["label"].numpy())
        num_occurences[label] += 1



    return num_occurences

train_occurences = count_occurences(train_ds, info)
print("train_ds:\t {}".format(train_occurences))

val_occurences = count_occurences(val_ds, info)
print("val_ds:\t\t {}".format(val_occurences))

test_occurences = count_occurences(test_ds, info)
print("test_ds:\t {}".format(test_occurences))


# your code
# Take exactly one item from the training dataset
one_item_ds = train_ds.take(1)

for item in one_item_ds:
    image = item["image"]   # image tensor

    print("Image dtype: {}".format(image.dtype))

    min_val = np.min(image)
    max_val = np.max(image)

    print("Smallest entry: {}".format(min_val))
    print("Largest entry: {}".format(max_val))


# your code
def preprocess(item):
    """
    Preprocess a single dataset element.

    Input:
        item: dict with keys "image" and "label"
    Output:
        (image_rescaled, label)
    """
    # Extract image and label from the dict
    image = item["image"]
    label = item["label"]

    # Convert image to float32 and scale to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0

    # Return a tuple (image, label) as required
    return image, label

train_ds = train_ds.map(preprocess)
val_ds   = val_ds.map(preprocess)
test_ds  = test_ds.map(preprocess)

batch_size = 32

train_ds = (
    train_ds
    .repeat()
    .shuffle(buffer_size=1024, seed=0)
    .batch(batch_size=batch_size)
    .prefetch(buffer_size=1)
)

# Validation dataset: batch size 32, no repeat/shuffle
val_ds = (
    val_ds
    .batch(batch_size=batch_size)
    .prefetch(buffer_size=1)
)

# Test dataset: batch size 1
test_ds = (
    test_ds
    .batch(batch_size=1)
    .prefetch(buffer_size=1)
)
# --------------------------

result_df = pd.read_csv("results.csv")

results_cnn = pd.DataFrame(columns=[
    'start_learning_rate', 'l2_weight', 'size_factor',
    'train_loss', 'val_loss', 'train_acc', 'val_acc'
])

# define hyperparameter search grid
start_learning_rates = [1e-5, 1e-4, 1e-3]
l2_weights = [0, 1e-5, 1e-4]
size_factors = [1, 2, 3]

# --------------------------
# START: Hyperparameter Loop
# --------------------------
epochs = 100
start_time_for_loop = time.time()

for start_learning_rate in start_learning_rates:
    for l2_weight in l2_weights:
        for size_factor in size_factors:

            # Build model
            model = keras.Sequential([
                keras.layers.Conv2D(
                    filters=6 * size_factor,
                    kernel_size=5,
                    activation='relu',
                    input_shape=info.features["image"].shape,
                    kernel_regularizer=keras.regularizers.l2(l2_weight)
                ),
                keras.layers.MaxPool2D(pool_size=2),

                keras.layers.Conv2D(
                    filters=16 * size_factor,
                    kernel_size=5,
                    activation='relu',
                    kernel_regularizer=keras.regularizers.l2(l2_weight)
                ),
                keras.layers.MaxPool2D(pool_size=2),

                keras.layers.Flatten(),

                keras.layers.Dense(
                    120 * size_factor,
                    activation='relu',
                    kernel_regularizer=keras.regularizers.l2(l2_weight)
                ),
                keras.layers.Dense(
                    84 * size_factor,
                    activation='relu',
                    kernel_regularizer=keras.regularizers.l2(l2_weight)
                ),

                keras.layers.Dense(
                    info.features["label"].num_classes,
                    activation='softmax'
                )
            ])

            # Learning rate schedule
            scheduler = keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=start_learning_rate,
                decay_steps=epochs * (np.sum(train_occurences) // batch_size),
                end_learning_rate=1e-8,
                power=1.0
            )

            # Callbacks
            stop_early = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )

            tensorboard_callback = keras.callbacks.TensorBoard(
                log_dir=f'logs/lr{start_learning_rate}_l2{l2_weight}_size{size_factor}'
            )

            # Compile
            model.compile(
                loss=keras.losses.sparse_categorical_crossentropy,
                optimizer=keras.optimizers.Adam(learning_rate=scheduler),
                metrics=[keras.metrics.sparse_categorical_accuracy]
            )

            # Train
            model.fit(
                train_ds,
                epochs=epochs,
                steps_per_epoch=np.sum(train_occurences) // batch_size,
                validation_data=val_ds,
                callbacks=[stop_early, tensorboard_callback],
                verbose=0
            )

            # Evaluate
            train_loss, train_acc = model.evaluate(
                train_ds, steps=np.sum(train_occurences) // batch_size, verbose=0
            )
            val_loss, val_acc = model.evaluate(val_ds, verbose=0)

            # Record results
            results_tmp = np.array([
                start_learning_rate, l2_weight, size_factor,
                train_loss, val_loss, train_acc, val_acc
            ]).reshape(1, -1)

            results_cnn = pd.concat(
                [results_cnn, pd.DataFrame(data=results_tmp, columns=results_cnn.columns)],
                axis=0
            )

            results_cnn.to_csv("results_cnn.csv", index=False)

end_time_for_loop = time.time()
print(f"Running time for loop: {end_time_for_loop - start_time_for_loop:.2f} seconds")

# --------------------------
# END SCRIPT
# --------------------------
