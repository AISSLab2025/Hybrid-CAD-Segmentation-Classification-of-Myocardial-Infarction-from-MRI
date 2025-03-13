# import the important libraries
import os
import numpy as np
import cv2
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import tensorflow_addons as tfa
from vit_keras import vit
from tensorflow.keras import Model # if this is not work, use the next commented one
#from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau , ModelCheckpoint
from collections import Counter
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.regularizers import l2

# models
from tensorflow.keras.applications import Xception,InceptionV3,InceptionResNetV2
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.vgg19 import VGG19
from livelossplot.inputs.keras import PlotLossesCallback
from tensorflow.keras.applications import DenseNet201, Xception
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from livelossplot import PlotLossesKeras
from tensorflow.keras.layers import AveragePooling2D,Dropout,Flatten,Dense,GlobalAveragePooling2D,BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import SGD, Adam
from livelossplot.inputs.keras import PlotLossesCallback
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,precision_recall_fscore_support
# Display gram-cam
from IPython.display import Image, display
import matplotlib.cm as mm1
import matplotlib as mm2

def load_dataset(data_train_path,data_test_path):
    train_datagen = ImageDataGenerator()
    test_datagen = ImageDataGenerator()
    val_datagen = ImageDataGenerator()
    training_set = train_datagen.flow_from_directory(data_train_path,
                                                 target_size = (image_size, image_size),
                                                 batch_size = BATCH_SIZE,
                                                 class_mode = 'categorical', shuffle=True)

    testing_set = test_datagen.flow_from_directory(data_test_path,
                                               target_size = (image_size, image_size),
                                               batch_size = BATCH_SIZE,
                                               class_mode = 'categorical', shuffle = False)
    return training_set, testing_set

def create_ResNet50_model(input_shape, n_classes, optimizer='rmsprop', fine_tune=0):
    """
    ResNet50 model
    """
    # Pretrained convolutional layers are loaded using the Imagenet weights.
    # Include_top is set to False, in order to exclude the model's fully-connected layers.
    conv_base = ResNet50(weights="imagenet", include_top=False,
    input_shape=input_shape)
    
    # Defines how many layers to freeze during training.
    # Layers in the convolutional base are switched from trainable to non-trainable
    # depending on the size of the fine-tuning parameter.
    for i in range(len(conv_base.layers)) :
        if i>=fine_tune:
            conv_base.layers[i].trainable = True
        else:
            conv_base.layers[i].trainable = False
    
    x = GlobalAveragePooling2D(name="avg_pool")(conv_base.output)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    top_dropout_rate = 0.5
    x = Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = Dense(n_classes, activation="softmax" )(x)
    
    # Compile
    model = Model(conv_base.input, outputs=outputs, name="ResNet50")

    # Compiles the model for training.
    #
    if n_classes==2:
        # for binary
        model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    else:
        # for multiclass
        model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    
    return model

def create_Xception_model(input_shape, n_classes, optimizer='rmsprop', fine_tune=0):
    """
    Xception model
    """
    # Pretrained convolutional layers are loaded using the Imagenet weights.
    # Include_top is set to False, in order to exclude the model's fully-connected layers.
    conv_base = Xception(weights="imagenet", include_top=False,
    input_shape=input_shape)
    
    # Defines how many layers to freeze during training.
    # Layers in the convolutional base are switched from trainable to non-trainable
    # depending on the size of the fine-tuning parameter.
    for i in range(len(conv_base.layers)) :
        if i>=fine_tune:
            conv_base.layers[i].trainable = True
        else:
            conv_base.layers[i].trainable = False
    
    x = GlobalAveragePooling2D(name="avg_pool")(conv_base.output)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    top_dropout_rate = 0.5
    x = Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = Dense(n_classes, activation="softmax" )(x)
    
    # Compile
    model = Model(conv_base.input, outputs=outputs, name="Xception")

    # Compiles the model for training.
    #
    if n_classes==2:
        # for binary
        model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    else:
        # for multiclass
        model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    
    return model

def create_InceptionResNetV2_model(input_shape, n_classes, optimizer='rmsprop', fine_tune=0):
    """
    InceptionResNetV2 model
    """
    # Pretrained convolutional layers are loaded using the Imagenet weights.
    # Include_top is set to False, in order to exclude the model's fully-connected layers.
    conv_base = InceptionResNetV2(weights="imagenet", include_top=False,
    input_shape=input_shape)
    
    # Defines how many layers to freeze during training.
    # Layers in the convolutional base are switched from trainable to non-trainable
    # depending on the size of the fine-tuning parameter.
    for i in range(len(conv_base.layers)) :
        if i>=fine_tune:
            conv_base.layers[i].trainable = True
        else:
            conv_base.layers[i].trainable = False
    
    x = GlobalAveragePooling2D(name="avg_pool")(conv_base.output)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    top_dropout_rate = 0.5
    x = Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = Dense(n_classes, activation="softmax" )(x)
    
    # Compile
    model = Model(conv_base.input, outputs=outputs, name="InceptionResNetV2")

    # Compiles the model for training.
    #
    if n_classes==2:
        # for binary
        model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    else:
        # for multiclass
        model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    
    return model

def create_ResNet50V2_model(input_shape, n_classes, optimizer='rmsprop', fine_tune=0):
    """
    ResNet50V2 model
    """
    # Pretrained convolutional layers are loaded using the Imagenet weights.
    # Include_top is set to False, in order to exclude the model's fully-connected layers.
    conv_base = ResNet50V2(weights="imagenet", include_top=False,
    input_shape=input_shape)
    
    # Defines how many layers to freeze during training.
    # Layers in the convolutional base are switched from trainable to non-trainable
    # depending on the size of the fine-tuning parameter.
    for i in range(len(conv_base.layers)) :
        if i>=fine_tune:
            conv_base.layers[i].trainable = True
        else:
            conv_base.layers[i].trainable = False
    
    x = GlobalAveragePooling2D(name="avg_pool")(conv_base.output)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    top_dropout_rate = 0.5
    x = Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = Dense(n_classes, activation="softmax" )(x)
    
    # Compile
    model = Model(conv_base.input, outputs=outputs, name="ResNet50V2")

    # Compiles the model for training.
    #
    if n_classes==2:
        # for binary
        model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    else:
        # for multiclass
        model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    
    return model

def create_DenseNet201_model(input_shape, n_classes, optimizer='rmsprop', fine_tune=0):
    """
    DenseNet201 model
    """
    # Pretrained convolutional layers are loaded using the Imagenet weights.
    # Include_top is set to False, in order to exclude the model's fully-connected layers.
    conv_base = DenseNet201(weights="imagenet", include_top=False,
    input_shape=input_shape)
    
    # Defines how many layers to freeze during training.
    # Layers in the convolutional base are switched from trainable to non-trainable
    # depending on the size of the fine-tuning parameter.
    for i in range(len(conv_base.layers)) :
        if i>=fine_tune:
            conv_base.layers[i].trainable = True
        else:
            conv_base.layers[i].trainable = False
    
    x = GlobalAveragePooling2D(name="avg_pool")(conv_base.output)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    top_dropout_rate = 0.5
    x = Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = Dense(n_classes, activation="softmax" )(x)
    
    # Compile
    model = Model(conv_base.input, outputs=outputs, name="DenseNet201")

    # Compiles the model for training.
    #
    if n_classes==2:
        # for binary
        model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    else:
        # for multiclass
        model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    
    return model

def create_VGG16_model(input_shape, n_classes, optimizer='rmsprop', fine_tune=0):
    """
    VGG16 model
    """
    # Pretrained convolutional layers are loaded using the Imagenet weights.
    # Include_top is set to False, in order to exclude the model's fully-connected layers.
    conv_base = VGG16(weights="imagenet", include_top=False,
    input_shape=input_shape)
    
    # Defines how many layers to freeze during training.
    # Layers in the convolutional base are switched from trainable to non-trainable
    # depending on the size of the fine-tuning parameter.
    for i in range(len(conv_base.layers)) :
        if i>=fine_tune:
            conv_base.layers[i].trainable = True
        else:
            conv_base.layers[i].trainable = False
    
    x = GlobalAveragePooling2D(name="avg_pool")(conv_base.output)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    top_dropout_rate = 0.5
    x = Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = Dense(n_classes, activation="softmax" )(x)
    
    # Compile
    model = Model(conv_base.input, outputs=outputs, name="VGG16")

    # Compiles the model for training.
    #
    if n_classes==2:
        # for binary
        model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    else:
        # for multiclass
        model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    
    return model

def ensemble1_model(model1_in,model2_in,n_classes,optimizer):
    model1 = Model(model1_in.input, model1_in.layers[-2].output)
    model2 = Model(model2_in.input, model2_in.layers[-2].output)
    for layer in model2.layers:
    	layer.trainable = False
    
    for layer in model1.layers:
    	layer.trainable = False
    
    input_1 = tf.keras.layers.Input((256, 256, 3))
    tower_1 = model1(input_1)
    tower_2 = model2(input_1)
    
    merge = tf.keras.layers.concatenate([tower_1,tower_2], name="concatallprobs")
    
    x = tf.keras.layers.Dense(1024, activation='relu')(merge)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    OUT = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
    
    Ensemble_model = Model(inputs=input_1, outputs=OUT)
    
    #optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=0.2), loss=categorical_smooth_loss,
    if n_classes==2:
        # for binary
        Ensemble_model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    else:
        # for multiclass
        Ensemble_model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

    return Ensemble_model

def ensemble2_model(model1_in,model2_inوmodel2_in,n_classes,optimizer):
    model1 = Model(model1_in.input, model1_in.layers[-2].output)
    model2 = Model(model2_in.input, model2_in.layers[-2].output)
    model3 = Model(model3_in.input, model3_in.layers[-2].output)

    for layer in model3.layers:
    	layer.trainable = False
    for layer in model2.layers:
    	layer.trainable = False
    for layer in model1.layers:
    	layer.trainable = False
    
    input_1 = tf.keras.layers.Input((256, 256, 3))
    tower_1 = model1(input_1)
    tower_2 = model2(input_1)
    tower_3 = model3(input_1)
    
    merge = tf.keras.layers.concatenate([tower_1,tower_2,tower_3], name="concatallprobs")
    
    x = tf.keras.layers.Dense(1024, activation='relu')(merge)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    OUT = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
    
    Ensemble_model = Model(inputs=input_1, outputs=OUT)
    
    #optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=0.2), loss=categorical_smooth_loss,
    if n_classes==2:
        # for binary
        Ensemble_model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    else:
        # for multiclass
        Ensemble_model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

    return Ensemble_model

def Ens_VIT(input_model,opt,n_classes):
    # ViT based CNN model(vgg16, resnet50,...)
    IMAGE_SIZE = 256
    top_dropout_rate = 0.5

    vit_model = vit.vit_b16(#vit_l32(
            image_size = IMAGE_SIZE,
            activation = 'softmax',
            pretrained = True,
            include_top = False,
            pretrained_top = False,
            classes = n_classes)

    inputs = tf.keras.layers.Input((IMAGE_SIZE, IMAGE_SIZE, 3))
    input_model_output = tf.keras.layers.Flatten()(input_model(inputs))
    vit_output = vit_model(inputs)
    
    x = tf.keras.layers.Concatenate(axis=-1)([input_model_output, vit_output], name="concatallprobs")
    
    x = tf.keras.layers.Dense(1024, activation = "relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)

    outputs = tf.keras.layers.Dense(n_classes, 'softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    
    if n_classes==2:
        # for binary
        model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    else:
        # for multiclass
        model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
    
    return model

#from livelossplot.keras import PlotLossesCallback
#from livelossplot import PlotLossesKeras

def save_weights(model_name):
    tl_checkpoint = ModelCheckpoint(filepath=model_name+'.hdf5',
                                    mode='max',monitor='val_accuracy',
                                  save_best_only=True,
                                  save_weights_only=True,
                                  verbose=1)
    plot_loss = PlotLossesCallback()
    # EarlyStopping
    early_stop = EarlyStopping(monitor='val_accuracy',
                           patience=50,
                           restore_best_weights=True,
                           mode='max'
                           #, min_delta=1
                          )
    return tl_checkpoint, early_stop, plot_loss 
# AUC for binary classification
def plot_roc_curve(fpr, tpr):
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='orange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
def draw_matrices(model,testing_set):
    Y_pred5 = model_res.predict(testing_set)
    Y_pred5 = np.argmax(Y_pred5, axis=1)
    print('Classification Report')
    print(classification_report(testing_set.classes, Y_pred5))

    conf_matrix = confusion_matrix(y_true=testing_set.classes, y_pred=Y_pred5)
    # Print the confusion matrix using Matplotlib
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Reds, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='large')
     
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()

    #calc other matrices  
    print('Accuracy: %.4f' % (accuracy_score(testing_set.classes, Y_pred5)))
    print('Recall ( Sen): %.4f' % (recall_score(testing_set.classes, Y_pred5,average = 'weighted')))
    print('Precision (Spe): %.4f' % (precision_score(testing_set.classes, Y_pred5,average = 'weighted')))
    print('F1 score : %.4f' % (f1_score(testing_set.classes, Y_pred5, average='weighted')))
    
    fpr, tpr, _ = roc_curve(testing_set.classes, Y_pred5)
    # calc AUC measurement
    print('AUC : %.4f' % (auc(fpr, tpr)))
    # draw AUC curve
    plot_roc_curve(fpr, tpr)

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1,2),keepdims=True)
    

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

import matplotlib.cm

def save_and_display_gradcam(img_path, heatmap, cam_path, alpha=0.4):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    #jet = mm2.colormaps.get_cmap("jet")
    jet = matplotlib.cm.get_cmap('jet')

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))

def get_layer_names(model):
    for layer in model.layers:
    if 'conv' not in layer.name:
        continue
    print(layer.name)
    
def train_model(model_name,training_set,testing_set):
    input_shape = (256, 256, 3)
    n_classes=5#2
    n_epochs = 100
    #optim_2 = Adam(learning_rate=0.0001)
    optim_2 = SGD(learning_rate=0.0001, momentum=0.9)
    preprocess_input = keras.applications.resnet50.preprocess_input
    decode_predictions = keras.applications.resnet50.decode_predictions
    
    if model_name=='resnet50':
        model=create_ResNet50_model(input_shape, n_classes,optim_2,  fine_tune=123)
        last_conv_layer_name = "conv5_block3_out" # use summary() of the model or us our get_layer_names() function
    elif model_name=='Xception':
        model=create_Xception_model(input_shape, n_classes,optim_2,  fine_tune=102) #fine_tune=702
        last_conv_layer_name = "block14_sepconv2"
    elif model_name=='InceptionResNetV2':
        model=create_InceptionResNetV2_model(input_shape, n_classes,optim_2,  fine_tune=106) #fine_tune=702
        last_conv_layer_name = "conv_7b" 
    elif model_name=='ResNet50V2':
        model=create_ResNet50V2_model(input_shape, n_classes,optim_2,  fine_tune=123)
        #preprocess_input = keras.applications.ResNet50V2.preprocess_input
        #decode_predictions = keras.applications.ResNet50V2.decode_predictions
        last_conv_layer_name = "conv5_block3_out"    
    elif model_name=='VGG16':
        model=create_VGG16_model(input_shape, n_classes,optim_2,  fine_tune=17) 
        preprocess_input = keras.applications.vgg16.preprocess_input
        decode_predictions = keras.applications.vgg16.decode_predictions
        last_conv_layer_name = "block5_conv3"
    elif model_name=='ensemble1':
        model1=create_ResNet50_model(input_shape, n_classes,optim_2,  fine_tune=123)
        model2=create_VGG16_model(input_shape, n_classes,optim_2,  fine_tune=17)
        model1.load_weights('resnet50.hdf5')
        model2.load_weights('VGG16.hdf5')
        model=ensemble1_model(model1,model2,n_classes,optim_2)
        last_conv_layer_name = "concatallprobs"
    elif model_name=='ensemble21':
        model1=create_ResNet50_model(input_shape, n_classes,optim_2,  fine_tune=123)
        model2=create_VGG16_model(input_shape, n_classes,optim_2,  fine_tune=17)
        model3=create_Xception_model(input_shape, n_classes,optim_2,  fine_tune=102)
        model1.load_weights('resnet50.hdf5')
        model2.load_weights('VGG16.hdf5')
        model3.load_weights('Xception.hdf5')
        model=ensemble2_model(model1,model2,model3,n_classes,optim_2)
        last_conv_layer_name = "concatallprobs"
    elif model_name=='ensemble22':
        model1=create_ResNet50_model(input_shape, n_classes,optim_2,  fine_tune=123)
        model2=create_VGG16_model(input_shape, n_classes,optim_2,  fine_tune=17)
        model3=create_ResNet50V2_model(input_shape, n_classes,optim_2,  fine_tune=123)
        model1.load_weights('resnet50.hdf5')
        model2.load_weights('VGG16.hdf5')
        model3.load_weights('ResNet50V2.hdf5')
        model=ensemble2_model(model1,model2,model3,n_classes,optim_2)
        last_conv_layer_name = "concatallprobs"
    elif model_name=='resnet50_vit':
        model1=create_ResNet50_model(input_shape, n_classes,optim_2,  fine_tune=123)
        model1.load_weights('resnet50.hdf5')
        model_resnet50 = Model(model1.input, model1.layers[-2].output) # remove classification layers
        for layer in model_resnet50.layers:
        	layer.trainable = False
        model=Ens_VIT(model_resnet50,optim_2,n_classes)
        last_conv_layer_name = "concatallprobs"
    elif model_name=='ens_vit1':
        model1=create_ResNet50_model(input_shape, n_classes,optim_2,  fine_tune=123)
        model2=create_VGG16_model(input_shape, n_classes,optim_2,  fine_tune=17)
        model_ensemble=ensemble1_model(model1,model2,n_classes,optim_2)
        model_ensemble.load_weights('ensemble1.hdf5')
        model_ens = Model(model_ensemble.input, model_ensemble.layers[-2].output) # remove classification layers
        for layer in model_ens.layers:
        	layer.trainable = False
        model=Ens_VIT(model_ens,optim_2,n_classes)
        last_conv_layer_name = "concatallprobs"
    elif model_name=='ens_vit21':
        model1=create_ResNet50_model(input_shape, n_classes,optim_2,  fine_tune=123)
        model2=create_VGG16_model(input_shape, n_classes,optim_2,  fine_tune=17)
        model3=create_Xception_model(input_shape, n_classes,optim_2,  fine_tune=102)
        model_ensemble=ensemble2_model(model1,model2,model3,n_classes,optim_2)
        model_ensemble.load_weights('ensemble21.hdf5')
        model_ens = Model(model_ensemble.input, model_ensemble.layers[-2].output) # remove classification layers
        for layer in model_ens.layers:
        	layer.trainable = False
        model=Ens_VIT(model_ens,optim_2,n_classes)
        last_conv_layer_name = "concatallprobs"
    elif model_name=='ens_vit22':
        model1=create_ResNet50_model(input_shape, n_classes,optim_2,  fine_tune=123)
        model2=create_VGG16_model(input_shape, n_classes,optim_2,  fine_tune=17)
        model3=create_ResNet50V2_model(input_shape, n_classes,optim_2,  fine_tune=123)
        model_ensemble=ensemble2_model(model1,model2,model3,n_classes,optim_2)
        model_ensemble.load_weights('ensemble22.hdf5')
        model_ens = Model(model_ensemble.input, model_ensemble.layers[-2].output) # remove classification layers
        for layer in model_ens.layers:
        	layer.trainable = False
        model=Ens_VIT(model_ens,optim_2,n_classes)
        last_conv_layer_name = "concatallprobs"

    
    # save the model automatically
    tl_checkpoint, early_stop, plot_loss = save_weights(model_name)
    history00 = model.fit(training_set, validation_data=testing_set,
                    callbacks=[tl_checkpoint,
                               early_stop, plot_loss],verbose=1, epochs=2)
    # load saved weight for the best results
    model.load_weights(model_name+'.hdf5')
    # draw matrices
    draw_matrices(model,testing_set)

    #Display  heat-map of the model
    img_size = (256, 256)
    img_path='C:/test/Positive/Case_P022_z006.png'
    # Prepare image
    img_array = preprocess_input(get_img_array(img_path, size=img_size))
    # Make model
    model_builder = model #(weights="imagenet")
    # Remove last layer's softmax
    model_builder.layers[-1].activation = None
    # Print what the top predicted class is
    preds = model_builder.predict(img_array)
    #print("Predicted:", decode_predictions(preds, top=1)[0])
    
    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    # Display heatmap
    plt.matshow(heatmap)
    plt.show()

    # save the heat-map into image
    save_and_display_gradcam(img_path, heatmap,model_name+'_heatmap.png')
