-#
# Created on 4/15/2021
#
# @author Seyed
#
# Email: mousavikahaki@gmail.com
#
import numpy as np
import tensorflow as tf
import segmentation_models as sm
import os, random, sys
import pandas as pd
import tqdm
import cv2
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras.losses import binary_crossentropy
from skimage.morphology import label

import warnings
warnings.filterwarnings('ignore')

Backbone_model = 'resnet34'

IMG_HEIGHT = 256
IMG_WIDTH = 256
N_CHANEL = 3

Batch_size = 8
Epochs = 10
Verbose = 1

TrainPath = 'stage1_train/'
TestPath = 'stage1_test/'

model_name = 'Unet_'+Backbone_model

def get_train_data(train_path):
    ids_train = next(os.walk(train_path))[1]
    n_train_img = len(ids_train)
    x_train = np.zeros((n_train_img, IMG_HEIGHT, IMG_WIDTH, N_CHANEL), dtype=np.uint8)
    y_train = np.zeros((n_train_img, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

    # for i, idx in tqdm_notebook(enumerate(ids_train), total=len(ids_train)):
    for i, idx in enumerate(ids_train):
        im_path = train_path + idx
        im = imread(im_path + '/images/' + idx + '.png')[:, :, :N_CHANEL]
        im = resize(im, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        x_train[i] = im

        label_max = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        ids_labels = next(os.walk(im_path + '/masks/'))[2]
        for label_file in ids_labels:
            label = imread(im_path + '/masks/' + label_file)
            label = resize(label, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
            label = np.expand_dims(label, axis=-1)
            label_max = np.maximum(label_max, label)

        y_train[i] = label_max

    return x_train, y_train


def get_test_data(test_path):
    ids_test = next(os.walk(test_path))[1]
    n_test_img = len(ids_test)
    x_test = np.zeros((n_test_img, IMG_HEIGHT, IMG_WIDTH, N_CHANEL), dtype=np.uint8)

    sizes_test = []
    # for i, idx in tqdm_notebook(enumerate(ids_test), total=len(ids_test)):
    for i, idx in enumerate(ids_test):
        im_path = test_path + idx
        im = imread(im_path + '/images/' + idx + '.png')[:, :, :N_CHANEL]
        im = resize(im, (IMG_WIDTH, IMG_HEIGHT), mode='constant', preserve_range=True)
        sizes_test.append([im.shape[0], im.shape[1]])
        x_test[i] = im

    return x_test, sizes_test


def visualize_random_train(x,y):
    # Visualize Random Train Data
    ix = random.randint(0, len(x))
    has_mask = y[ix].max() > 0  # salt indicator

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 15))

    ax1.imshow(x[ix, ..., 0], cmap='gray', interpolation='bilinear')
    if has_mask:  # if salt
        # draw a boundary(contour) in the original image separating salt and non-salt areas
        ax1.contour(y[ix].squeeze(), colors='c', linewidths=5, levels=[0.5])
    ax1.set_title('Original')

    ax2.imshow(y[ix].squeeze(), cmap='gray', interpolation='bilinear')
    ax2.set_title('Label')


# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2, y_true)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    return 0.5 * binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)


X_test, sizes_test = get_test_data(TestPath)
X_train, Y_train = get_train_data(TrainPath)



visualize_random_train(X_train,Y_train)

# Pre_process images
pre_process_fn = sm.get_preprocessing(Backbone_model)
X_train = pre_process_fn(X_train)

X_train, X_val, Y_train, Y_val = train_test_split(X_train,Y_train, test_size=0.2, random_state=42)

model = sm.Unet(Backbone_model,encoder_weights='imagenet')
model.compile(optimizer='adam',loss=sm.losses.bce_jaccard_loss,metrics=[sm.metrics.iou_score])

# print(model.summary())
plot_model(model, to_file=model_name+'.png', show_shapes=True, show_layer_names=True)

callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model-tgs-salt.h5', verbose=1, save_best_only=True, save_weights_only=True)
]

history = model.fit(X_train,
                    Y_train,
                    batch_size=Batch_size,
                    epochs=Epochs,
                    verbose=Verbose,
                    validation_data=(X_val,Y_val),
                    callbacks=callbacks)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(loss) + 1)
plt.plot(epochs,loss,'y',label='Training Loss')
plt.plot(epochs,val_loss,'r',label='Validation Loss')
plt.plot( np.argmin(history.history["val_loss"]), np.min(history.history["val_loss"]), marker="x", color="r", label="best model")
plt.title('Learning curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.save(model_name + '.h5')


print('Done!')


# model = tf.keras.models.load_model(model_path)
#
# testIM = X_test[0]
# testIM = np.expand_dims(testIM,axis=0)
# prediction = model.predict(testIM)
#
# prediction_img = prediction.reshape(IMG_HEIGHT, IMG_WIDTH, 1)
# plt.imshow(prediction_img,cmap='gray')




# model.load_weights(model_name + '.h5')
# model.evaluate(X_val, Y_val, verbose=1)
#
# preds_train = model.predict(X_train, verbose=1)
# preds_val = model.predict(X_val, verbose=1)
#
# preds_train_t = (preds_train > 0.5).astype(np.uint8)
# preds_val_t = (preds_val > 0.5).astype(np.uint8)
#
#
# def plot_sample(X, y, preds, binary_preds, ix=None):
#     """Function to plot the results"""
#     if ix is None:
#         ix = random.randint(0, len(X))
#
#     has_mask = y[ix].max() > 0
#
#     fig, ax = plt.subplots(1, 4, figsize=(20, 10))
#     ax[0].imshow(X[ix, ..., 0], cmap='gray')
#     if has_mask:
#         ax[0].contour(y[ix].squeeze(), colors='r', levels=[0.5])
#     ax[0].set_title('Original')
#
#     ax[1].imshow(y[ix].squeeze(), cmap='gray')
#     ax[1].set_title('Label')
#
#     ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1, cmap='gray')
#     if has_mask:
#         ax[2].contour(y[ix].squeeze(), colors='r', levels=[0.5])
#     ax[2].set_title('Predicted')
#
#     ax[3].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1, cmap='gray')
#     if has_mask:
#         ax[3].contour(y[ix].squeeze(), colors='r', levels=[0.5])
#     ax[3].set_title('Predicted binary')
#
# plot_sample(X_train, Y_train, preds_train, preds_train_t)


# # For Kaggle submission
# def rle_encoding(x):
#     dots = np.where(x.T.flatten() == 1)[0]
#     run_lengths = []
#     prev = -2
#     for b in dots:
#         if (b>prev+1): run_lengths.extend((b + 1, 0))
#         run_lengths[-1] += 1
#         prev = b
#     return run_lengths
#
# def prob_to_rles(x, cutoff=0.5):
#     lab_img = label(x > cutoff)
#     for i in range(1, lab_img.max() + 1):
#         yield rle_encoding(lab_img == i)
#
# preds_test = model.predict(X_test, verbose=1)
#
# preds_test_upsampled = []
# for i in range(len(preds_test)):
#     preds_test_upsampled.append(cv2.resize(preds_test[i],
#                                            (sizes_test[i][1], sizes_test[i][0])))
#
# test_ids = next(os.walk(TestPath))[1]
# new_test_ids = []
# rles = []
# for n, id_ in enumerate(test_ids):
#     rle = list(prob_to_rles(preds_test_upsampled[n]))
#     rles.extend(rle)
#     new_test_ids.extend([id_] * len(rle))
#
# sub = pd.DataFrame()
# sub['ImageId'] = new_test_ids
# sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
# sub.to_csv('submission_relu.csv', index=False)



# IM1 = preds_test[0,:,:,0]
# plt.imshow(IM1,cmap='gray')
# plt.show()

# ## For Kaggle 2
# import skimage
# preds_test= model.predict(X_test, verbose=1)
# def return_shape_masks(masks, shapes):
#     reshape_mask = []
#     for i, shape in enumerate(shapes):
#         mask = skimage.transform.resize(
#             masks[i],
#             shape,
#             mode='reflect')
#         mask = (mask > 0.5).astype(np.uint8)
#         reshape_mask.append(mask)
#     return np.array(reshape_mask)
#
# def return_shape_img(images, shapes):
#     reshape_images = []
#     for i, shape in enumerate(shapes):
#         img = skimage.transform.resize(
#             images[i],
#             (shape[0], shape[1], 3),
#             mode='reflect')
#         reshape_images.append(img)
#     return np.array(reshape_images)
# from skimage.morphology import watershed
# from scipy import ndimage as ndi
# from skimage.feature import peak_local_max
#
# def rle_encoding(x): # функция находит все точки на изображении
#     '''x: numpy array of shape (height, width), 1 - mask, 0 - background
#     Returns run length as list'''
#     dots = np.where(x.T.flatten() > 0.5)[0] # .T sets Fortran order down-then-right
#     run_lengths = []
#     prev = -2
#     for b in dots:
#         if (b>prev+1): run_lengths.extend((b+1, 0))
#         run_lengths[-1] += 1
#         prev = b
#     return run_lengths
#
# def prob_to_rles(x, cutoff=0.50): # функция разделяет пятна на группы
#     lab_img = label(x > cutoff)
#     for i in range(1, lab_img.max() + 1):
#         yield rle_encoding(lab_img == i)
#
# input_shape = [img.shape[:2] for img in X_test]
# images = return_shape_img(X_test, input_shape)
# masks = return_shape_masks(preds_test, input_shape)
#
# for i, mask in enumerate(masks[:10]):
#     plt.figure(figsize = (14, 5))
#     plt.subplot(1, 2, 1); plt.imshow(images[i])
#     plt.subplot(1, 2, 2); plt.imshow(mask[:,:,0])
#     plt.show()
#
# new_test_ids = []
# rles = []
# test_img_name = [img for img in next(os.walk('stage1_test/'))[1]]
# for i, id in enumerate(test_img_name):
#     rle = list(prob_to_rles(masks[i][:,:,0]))
#     rles.extend(rle)
#     new_test_ids.extend([id] * len(rle))
#
# submission_df = pd.DataFrame()
# submission_df['ImageId'] = new_test_ids
# submission_df['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
# submission_df.to_csv('submission.csv', index=False)