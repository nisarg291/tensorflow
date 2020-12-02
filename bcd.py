import json
import math
import pickle
import os
import cv2
from PIL import Image
import numpy as np
from keras import layers
from keras.applications import DenseNet201
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score
import scipy
from tqdm import tqdm
import tensorflow as tf
from keras import backend as K
import gc
from functools import partial
from sklearn import metrics
from collections import Counter
import json
import itertools

# Next I loaded the images in the respective folders.

def Dataset_loader(DIR, RESIZE, sigmaX=10):
    IMG = []
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    for IMAGE_NAME in tqdm(os.listdir(DIR)):
        PATH = os.path.join(DIR,IMAGE_NAME)
        _, ftype = os.path.splitext(PATH)
        if ftype == ".png":
            img = read(PATH)
           
            img = cv2.resize(img, (RESIZE,RESIZE))
           
            IMG.append(np.array(img))
    return IMG

benign_train = np.array(Dataset_loader('Dataset_BUSI/train/benign',224))
malign_train = np.array(Dataset_loader('Dataset_BUSI/train/malignant',224))
normal_train= np.array(Dataset_loader('Dataset_BUSI/train/normal',224))
benign_test = np.array(Dataset_loader('Dataset_BUSI/test/benign',224))
malign_test = np.array(Dataset_loader('Dataset_BUSI/test/malignant',224))
normal_test= np.array(Dataset_loader('Dataset_BUSI/test/normal',224))
# After that I created a numpy array of zeroes for labeling benign images and similarly a numpy array of ones for labeling malignant images.
#  I also shuffled the dataset and converted the labels into categorical format.

benign_train_label = np.zeros(len(benign_train))
malign_train_label = np.ones(len(malign_train))
benign_test_label = np.zeros(len(benign_test))
malign_test_label = np.ones(len(malign_test))

X_train = np.concatenate((benign_train, malign_train), axis = 0)
Y_train = np.concatenate((benign_train_label, malign_train_label), axis = 0)
X_test = np.concatenate((benign_test, malign_test), axis = 0)
Y_test = np.concatenate((benign_test_label, malign_test_label), axis = 0)

s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train = X_train[s]
Y_train = Y_train[s]

s = np.arange(X_test.shape[0])
np.random.shuffle(s)
X_test = X_test[s]
Y_test = Y_test[s]

Y_train = to_categorical(Y_train, num_classes= 2)
Y_test = to_categorical(Y_test, num_classes= 2)

# Then I split the data-set into two sets — train and test sets with 80% and 20% images respectively.
#  Let’s see some sample benign and malignant images.


x_train, x_val, y_train, y_val = train_test_split(
    X_train, Y_train, 
    test_size=0.2, 
    random_state=11
)

w=60
h=40
fig=plt.figure(figsize=(15, 15))
columns = 4
rows = 3

for i in range(1, columns*rows +1):
    ax = fig.add_subplot(rows, columns, i)
    if np.argmax(Y_train[i]) == 0:
        ax.title.set_text('Benign')
    else:
        ax.title.set_text('Malignant')
    plt.imshow(x_train[i], interpolation='nearest')
plt.show()

# I used a batch size value of 16. 
# Batch size is one of the most important hyperparameters to tune in deep learning.
#  I prefer to use a larger batch size to train my models as it allows computational speedups from the parallelism of GPUs. 
# However, it is well known that too large of a batch size will lead to poor generalization. 
# On the one extreme, using a batch equal to the entire dataset guarantees convergence to the global optima of the objective function.
#  However this is at the cost of slower convergence to that optima.
#  On the other hand, using smaller batch sizes have been shown to have faster convergence to good results.
#  This is intuitively explained by the fact that smaller batch sizes allow the model to start learning before having to see all the data. 
# The downside of using a smaller batch size is that the model is not guaranteed to converge to the global optima.
# Therefore it is often advised that one starts at a small batch size reaping the benefits of faster training dynamics and steadily grows the batch size through training.

# I also did some data augmentation. 
# The practice of data augmentation is an effective way to increase the size of the training set. 
# Augmenting the training examples allow the network to see more diversified, but still representative data points during training.

# Then I created a data generator to get the data from our folders and into Keras in an automated way. 
# Keras provides convenient python generator functions for this purpose.

BATCH_SIZE = 16

train_generator = ImageDataGenerator(
        zoom_range=2,  # set range for random zoom
        rotation_range = 90,
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
    )

# The next step was to build the model. This can be described in the following 3 steps:

# I used DenseNet201 as the pre trained weights which is already trained in the Imagenet competition. The learning rate was chosen to be 0.0001.
# On top of it I used a globalaveragepooling layer followed by 50% dropouts to reduce over-fitting.
# I used batch normalization and a dense layer with 2 neurons for 2 output classes ie benign and malignant with softmax as the activation function.
# I have used Adam as the optimizer and binary-cross-entropy as the loss function.

def build_model(backbone, lr=1e-4):
    model = Sequential()
    model.add(backbone)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(2, activation='softmax'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=lr),
        metrics=['accuracy']
    )
    return model

resnet = DenseNet201(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)

model = build_model(resnet ,lr = 1e-4)
print(model.summary())

# Let’s see the output shape and the parameters involved in each layer.

# def build_model(backbone, lr=1e-4):
#     model = Sequential()
#     model.add(backbone)
#     model.add(layers.GlobalAveragePooling2D())
#     model.add(layers.Dropout(0.5))
#     model.add(layers.BatchNormalization())
#     model.add(layers.Dense(2, activation='softmax'))
    
#     model.compile(
#         loss='binary_crossentropy',
#         optimizer=Adam(lr=lr),
#         metrics=['accuracy']
#     )
#     return model

# resnet = DenseNet201(
#     weights='imagenet',
#     include_top=False,
#     input_shape=(224,224,3)
# )

# model = build_model(resnet ,lr = 1e-4)
# model.summary()

learn_control = ReduceLROnPlateau(monitor='val_accuracy', patience=0,
                                  verbose=1,factor=0.2, min_lr=1e-7)

filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

history = model.fit_generator(
    train_generator.flow(x_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=x_train.shape[0] / BATCH_SIZE,
    epochs=10,
    validation_data=(x_val, y_val),
    callbacks=[learn_control, checkpoint]
)

# Performance Metrics The most common metric for evaluating model performance is the accurcacy.
#  However, when only 2% of your dataset is of one class (malignant) and 98% some other class (benign), misclassification scores don’t really make sense.
#  You can be 98% accurate and still catch none of the malignant cases which could make a terrible classifier.

history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot()

history_df = pd.DataFrame(history.history)
history_df[['accuracy', 'val_accuracy']].plot()

filename='final_model.sav'
pickle.dump(model, open(filename, 'wb'))

# Confusion Matrix
 
# Confusion Matrix is a very important metric when analyzing misclassification. 
# Each row of the matrix represents the instances in a predicted class while each column represents the instances in an actual class. 
# The diagonals represent the classes that have been correctly classified. 
# This helps as we not only know which classes are being misclassified but also what they are being misclassified as.

from sklearn.metrics import classification_report
classification_report( np.argmax(Y_test, axis=1), np.argmax(Y_pred_tta, axis=1))

from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=55)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

cm = confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(Y_pred, axis=1))

cm_plot_label =['benign', 'malignant']
plot_confusion_matrix(cm, cm_plot_label, title ='Confusion Metrix for Skin Cancer')

# ROC Curves
 
# The 45 degree line is the random line, where the Area Under the Curve or AUC is 0.5 . 
# The further the curve from this line, the higher the AUC and better the model. 
# The highest a model can get is an AUC of 1, where the curve forms a right angled triangle. 
# The ROC curve can also help debug a model. 
# For example, if the bottom left corner of the curve is closer to the random line, it implies that the model is misclassifying at Y=0.
#  Whereas, if it is random on the top right, it implies the errors are occurring at Y=1.

from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import roc_curve
roc_log = roc_auc_score(np.argmax(Y_test, axis=1), np.argmax(Y_pred_tta, axis=1))
false_positive_rate, true_positive_rate, threshold = roc_curve(np.argmax(Y_test, axis=1), np.argmax(Y_pred_tta, axis=1))
area_under_curve = auc(false_positive_rate, true_positive_rate)

plt.plot([0, 1], [0, 1], 'r--')
plt.plot(false_positive_rate, true_positive_rate, label='AUC = {:.3f}'.format(area_under_curve))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
#plt.savefig(ROC_PLOT_FILE, bbox_inches='tight')
plt.close()