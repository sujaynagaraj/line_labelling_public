from hospital_api import *
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D, AveragePooling1D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import *
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl
import keras_tuner as kt
import plotly.express as px

from os import listdir
from os.path import isfile, join

class MyHyperModel(kt.HyperModel):

    def __init__(self, size):
        self.size = size

    def build(self, hp):
        model = Sequential()
        hp_filters = hp.Int('filters', min_value=32, max_value=128, step=32)
        hp_filters_a = hp.Int('kernels_a', min_value=100, max_value=250, step=25)
        hp_filters_b = hp.Int('kernels_b', min_value=5, max_value=20, step=5)
        hp_dropouts = hp.Float('dropouts', min_value=0.10, max_value=0.5, step=0.05)
        hp_units = hp.Int('units', min_value=32, max_value=128, step=32)

        model.add(Conv1D(filters=hp_filters, kernel_size=hp_filters_a,strides=5,  activation='relu', input_shape=(self.size,1), data_format="channels_last"))
        model.add(Conv1D(filters=hp_filters, kernel_size=hp_filters_b,strides=5,  activation="relu"))
        model.add(BatchNormalization(axis=-1))
        model.add(Dropout(hp_dropouts))
        model.add(AveragePooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(hp_units, activation='relu'))
        model.add(Dropout(hp_dropouts))
        model.add(Dense(1, activation='sigmoid'))

        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        adam=keras.optimizers.Adam(learning_rate=hp_learning_rate)
        model.compile(loss='binary_crossentropy',
                    optimizer=adam,
                    metrics=[
                        keras.metrics.TruePositives(name='tp'),
                        keras.metrics.FalsePositives(name='fp'),
                        keras.metrics.TrueNegatives(name='tn'),
                        keras.metrics.FalseNegatives(name='fn'),
                        keras.metrics.BinaryAccuracy(name='accuracy'),
                        keras.metrics.Precision(name='precision'),
                        keras.metrics.Recall(name='recall'),
                        keras.metrics.AUC(name='auc')
                        ])
        return model

    

def load_dataset(path):
    with open(path, "rb") as f:
        x_train, y_train, x_val, y_val, x_test, y_test = pkl.load(f)

    x_train=np.array(x_train)
    y_train=np.array(y_train)
    x_val=np.array(x_val)
    y_val=np.array(y_val)
    x_test=np.array(x_test)
    y_test=np.array(y_test)


    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1] , 1)
    x_val = x_val.reshape(x_val.shape[0],x_val.shape[1] , 1)
    x_test = x_test.reshape(x_test.shape[0],x_test.shape[1] , 1)

    return x_train, y_train, x_val, y_val, x_test, y_test


def get_tuner(size, name):
    # Instantiate the tuner
    my_hyper_model = MyHyperModel(size = size)
    tuner = kt.Hyperband(my_hyper_model, # the hypermodel
                        objective=kt.Objective("val_auc", direction="max"), # objective to optimize
    max_epochs=10,
    factor=3, # factor which you have seen above 
    directory="logs/abp", # directory to save logs 
    project_name=name)

    # hypertuning settings
    #print(tuner.search_space_summary())

    
    return tuner

def model_eval(model, x_test, y_test, file):
    y_pred=(model.predict(x_test) > 0.5).astype("int32")
    
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_test, y_pred)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test, y_pred)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, y_pred)
    print('F1 score: %f' % f1)
    roc_auc=roc_auc_score(y_test, y_pred)
    print('ROC AUC score: %f' % roc_auc)

    plt.figure()
    sns.heatmap(confusion_matrix(y_test, y_pred), square=True, annot=True, fmt='d', cbar=False,
                xticklabels=["Normal","Artefact"],
             yticklabels=["Normal","Artefact"])
    plt.xlabel('predicted label')
    plt.ylabel('true label')

    save_path = "results/abp/confusion/"+file[:-4]+".png"

    plt.savefig(save_path)


    # # predict probabilities
    probs = model.predict(x_test)
    fpr, tpr, thresholds = roc_curve(y_test, probs)

    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    #fig.show()

    save_path = "results/abp/ROC/"+file[:-4]+".png"
    fig.write_image(save_path)

def main():
    #get list of files in datasets/ path
    files = [f for f in listdir("datasets/abp/") if isfile(join("datasets/abp/", f))]

    for file in files:
        #Load Dataset
        x_train, y_train, x_val, y_val, x_test, y_test = load_dataset("datasets/abp/"+file)
        if (len(np.unique(y_train)) !=2) or  (len(np.unique(y_val)) !=2) or (len(np.unique(y_test)) !=2):
            continue

        window_size = int(file.split('_')[-1][:-4]) #string parsing to get the right window size
        file_short = file[:-4]

        #Load Tuner
        tuner = get_tuner(window_size, file_short)
        stop_early = EarlyStopping(monitor='val_loss', patience=5)
        tuner.search(x_train, y_train, epochs=10, validation_data=(x_val, y_val), callbacks=[stop_early])

        # Get the optimal hyperparameters
        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

        # Build the model with the optimal hyperparameters and train it on the data for 200 epochs
        model = tuner.hypermodel.build(best_hps)
        checkpoint_path = "best_weights/abp/" +file_short + "_weights.h5"

        checkpoint = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss', verbose=0, mode='min')

        history = model.fit(x_train, y_train, epochs=200, validation_data=(x_val, y_val), callbacks = checkpoint)
        
        model.load_weights(checkpoint_path)

        #MODEL EVAL AND SAVE RESULTS
        model_eval(model, x_test, y_test, file)

        #SAVE MODEL
        save_path = "models/abp/"+file_short+".h5"

        model.save(save_path)

if __name__ == "__main__":
    main()