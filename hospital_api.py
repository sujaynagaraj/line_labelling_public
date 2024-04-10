import pandas as pd
import io
import random
import requests
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle
from seglearn.feature_functions import *
from seglearn.transform import *

import matplotlib.pyplot as plt
import pickle as pkl
import gzip
from tqdm import tqdm



#Generates a panda dataframe with an API URL
def genDF (url):
    csv=requests.get(url).content
    df=pd.read_csv(io.StringIO(csv.decode('utf-8')), names=["Time", "Pressure"])
    df["Pressure"]=df["Pressure"].apply(convertmmhg)
    return df

#formula to convert integer representation of pressure to mmHg
def convertmmhg(val):
    return (val*0.0625-40)

#plots a dataframe
def plotter (df):
    df.plot("Time", "Pressure")
def plotter_bkpts (df, bkpts):
    ax = df.plot("Time", "Pressure")
    ax.vspan(x=bkpts,ymin=0,ymax=1, color='r', alpha=0.3)

def plotter_bkpts2 (df, bkpts_start, bkpts_end):
    ax = df.plot("Time", "Pressure")
    for i in range(len(bkpts_start)):
        ax.axvspan(xmin=bkpts_start[i],xmax=bkpts_end[i],ymin=0,ymax=1, facecolor='r', alpha=0.3)

#generates a DF based on a specificed interval_ID from artefact_summary.csv
def find_interval(interval_ID):
    summarydata = pd.read_csv('artefact_summary.csv')
    url = summarydata.loc[summarydata["interval_ID"] == interval_ID, "URL"].iloc[0]
    url=url+"&format=csv"
    return genDF(url)

#generates a pd DF from the artefact_summary.csv file
def artefact_summary():
    summarydata = pd.read_csv('artefact_summary.csv')
    #drops one of the negative duration ones - error entry interval_ID 164 which is index 162
    summarydata=summarydata.drop(162)
    return summarydata

#takes list and returns [lists] of random, contiguous, non-overlapping subsequences
def rand_windows(lis, num, length):
    indices=range(len(lis)-(length-1)*num)
    outs=[]
    offset=0
    for i in sorted(random.sample(indices,num)):
        i += offset
        outs.append(lis[i:i+length])
        offset += length-1
    return outs

def rand_windows2(lis, num, length):
    if length>len(lis):
        outs=[]
        outs.append(lis)
        return outs
    elif num*length>len(lis):
        maxseg=(len(lis))//(length)
        return rand_windows2(lis,maxseg,length)
    else:
        indices=range(len(lis)-(length-1)*num)
        outs=[]
        offset=0
        for i in sorted(random.sample(indices,num)):
            i += offset
            outs.append(lis[i:i+length])
            offset += length-1
        return outs

#DEPRACATED VERSION
#returns list of indices instead of lists
""" def rand_windows3(lis, num, start, length):
    if length>len(lis[start:]):
        outs=[]
        outs.append(lis)
        return outs
    elif num*length>len(lis[start:]):
        maxseg=(len(lis[start:]))//(length)
        return rand_windows3(lis,maxseg,start, length)
    else:
        indices=range(start, len(lis)-(length-1)*num)
        outs=[]
        offset=0
        for i in sorted(random.sample(indices,num)):
            i += offset
            tup = (i,i+length)
            outs.append(tup)
            offset += length-1
        return outs """

#Given a waveform, number of slices to extract, an indice of when to start extracting (ie: after scaling buffer), and a desired
#length of slices. returns random slices after leaving a specified length scaling buffer
def rand_windows3(lis, num, start, length):
    outs=[]
    max=(len(lis[start:]))//(length) #Max number of slices we can extract (after scaling buffer)

    num = min(num, max)

    if num == 0:
        return [(len(lis)-length-1,len(lis)-1)]
    elif num == 1: ##IF DESIRED WINDOW SIZE > DATA (excluding the scaling buffer of 10 min)
                                            ## OR IF ONLY ONE SLICE CAN BE EXTRACTED
        return [(start, start+length)]
    else:
        indices=range(start, len(lis)-(length-1)*num)

        offset=0
        for i in sorted(random.sample(indices,num)):
            i += offset
            tup = (i,i+length)
            outs.append(tup)
            offset += length-1
        return outs

#creates a training data set with complex labels (start, during and end of artefact) with size of vectors you want out
# [X][label]
#number of training examples to draw from
#size of vectors you want out
def complex_labels2(number, size):
    data= pd.read_csv("label_00002.csv", names=["Time", "Integer", "Pressure"])
    data = data.drop(columns=["Integer"])
    data["Time"] = data["Time"].apply(lambda x: (x + 599.816))

    #y_labels = []

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    data.loc[:, :] = scaled

    data = data.drop(columns="Time")
    data = data.values.tolist()
    data = [val for sublist in data for val in sublist]

    X=rand_windows(data, number, size)
    return X

#def complex_labels_rand(number, size):

#generate a ~5D vector with mean, median, max, min, std
#def simple_labels():


def gen_labels(samples, num, size):
    summarydata=artefact_summary()
    summarydata = summarydata.sample(samples)
    X = []
    y_labels = []

    #Half normal half artefact
    num=num//2

    #normal
    for index, row in summarydata.iterrows():
        device = str(row["deviceID"])
        start = str(row["interval_start"])
        end = str(row["artifact_start"])
        url = str(genURL(device, start, end))
        df=genDF(url)

        #normalizing
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df)
        df.loc[:, :] = scaled
        df = df.drop(columns="Time")
        df = df.values.tolist()
        df = [val for sublist in df for val in sublist]
        rand=rand_windows2(df, num, size)
        for lis in rand:
            X.append(lis)
            y_labels.append(0)

    #artefact
    for index, row in summarydata.iterrows():
        device = str(row["deviceID"])
        start = str(row["artifact_start"])
        end = str(row["artifact_finish"])
        url = str(genURL(device, start, end))
        artifactDF=genDF(url)

        #normalizing
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(artifactDF)
        artifactDF.loc[:, :] = scaled

        artifactDF = artifactDF.drop(columns="Time")
        artifactDF = artifactDF.values.tolist()
        artifactDF = [val for sublist in artifactDF for val in sublist]
        rand=rand_windows2(artifactDF, num, size)
        for lis in rand:
            X.append(lis)
            y_labels.append(1)

    X, y_labels = shuffle (X, y_labels)
    return X, y_labels

#no min/max scaling
def gen_labels2(samples, num, size):
    summarydata=artefact_summary()
    summarydata = summarydata.sample(samples)
    X = []
    y_labels = []

    #Half normal half artefact
    num=num//2

    #normal
    for index, row in summarydata.iterrows():
        device = str(row["deviceID"])
        start = str(row["interval_start"])
        end = str(row["artifact_start"])
        url = str(genURL(device, start, end))
        df=genDF(url)

        df = df.drop(columns="Time")
        df = df.values.tolist()
        df = [val for sublist in df for val in sublist]
        rand=rand_windows2(df, num, size)
        for lis in rand:
            X.append(lis)
            y_labels.append(0)

    #artefact
    for index, row in summarydata.iterrows():
        device = str(row["deviceID"])
        start = str(row["artifact_start"])
        end = str(row["artifact_finish"])
        url = str(genURL(device, start, end))
        artifactDF=genDF(url)

        artifactDF = artifactDF.drop(columns="Time")
        artifactDF = artifactDF.values.tolist()
        artifactDF = [val for sublist in artifactDF for val in sublist]
        rand=rand_windows2(artifactDF, num, size)
        for lis in rand:
            X.append(lis)
            y_labels.append(1)

    X, y_labels = shuffle (X, y_labels)
    return X, y_labels

#preventing data leakage, by ensuring segments from the same artefact sample aren't in the same dataset
#generates x_train, y_train, x_val, y_val, x_test, y_test
def gen_labels3(num, size):
    summarydata=artefact_summary()
    summarydata = summarydata.sample(1700) #random sampling
    train=summarydata[0:int(0.6*len(summarydata))]
    val=summarydata[int(0.6*len(summarydata)):int(0.8*len(summarydata))]
    test=summarydata[int(0.8*len(summarydata)):len(summarydata)]
    x_train=[]
    y_train=[]
    x_val=[]
    y_val=[]
    x_test=[]
    y_test=[]

    #normal
    for index, row in train.iterrows():
        device = str(row["deviceID"])
        start = str(row["interval_start"])
        end = str(row["artifact_start"])
        url = str(genURL(device, start, end))
        df=genDF(url)

        df = df.drop(columns="Time")
        df = df.values.tolist()
        df = [val for sublist in df for val in sublist]
        rand=rand_windows2(df, num, size)
        for lis in rand:
            x_train.append(lis)
            y_train.append(0)
    for index, row in val.iterrows():
        device = str(row["deviceID"])
        start = str(row["interval_start"])
        end = str(row["artifact_start"])
        url = str(genURL(device, start, end))
        df=genDF(url)

        df = df.drop(columns="Time")
        df = df.values.tolist()
        df = [val for sublist in df for val in sublist]
        rand=rand_windows2(df, num, size)
        for lis in rand:
            x_val.append(lis)
            y_val.append(0)
    for index, row in test.iterrows():
        device = str(row["deviceID"])
        start = str(row["interval_start"])
        end = str(row["artifact_start"])
        url = str(genURL(device, start, end))
        df=genDF(url)

        df = df.drop(columns="Time")
        df = df.values.tolist()
        df = [val for sublist in df for val in sublist]
        rand=rand_windows2(df, num, size)
        for lis in rand:
            x_test.append(lis)
            y_test.append(0)

    #artefact
    for index, row in train.iterrows():
        device = str(row["deviceID"])
        start = str(row["artifact_start"])
        end = str(row["artifact_finish"])
        url = str(genURL(device, start, end))
        artifactDF=genDF(url)

        artifactDF = artifactDF.drop(columns="Time")
        artifactDF = artifactDF.values.tolist()
        artifactDF = [val for sublist in artifactDF for val in sublist]
        rand=rand_windows2(artifactDF, num, size)
        for lis in rand:
            x_train.append(lis)
            y_train.append(1)
    for index, row in val.iterrows():
        device = str(row["deviceID"])
        start = str(row["artifact_start"])
        end = str(row["artifact_finish"])
        url = str(genURL(device, start, end))
        artifactDF=genDF(url)

        artifactDF = artifactDF.drop(columns="Time")
        artifactDF = artifactDF.values.tolist()
        artifactDF = [val for sublist in artifactDF for val in sublist]
        rand=rand_windows2(artifactDF, num, size)
        for lis in rand:
            x_val.append(lis)
            y_val.append(1)
    for index, row in test.iterrows():
        device = str(row["deviceID"])
        start = str(row["artifact_start"])
        end = str(row["artifact_finish"])
        url = str(genURL(device, start, end))
        artifactDF=genDF(url)

        artifactDF = artifactDF.drop(columns="Time")
        artifactDF = artifactDF.values.tolist()
        artifactDF = [val for sublist in artifactDF for val in sublist]
        rand=rand_windows2(artifactDF, num, size)
        for lis in rand:
            x_test.append(lis)
            y_test.append(1)

    errors=[]
    for i in range(len(x_train)):
        if len(x_train[i])!=size:
            errors.append(i)
    for elem in sorted(errors, reverse=True):
        del x_train[elem]
        del y_train[elem]
    errors=[]
    for i in range(len(x_val)):
        if len(x_val[i])!=size:
            errors.append(i)
    for elem in sorted(errors, reverse=True):
        del x_val[elem]
        del y_val[elem]
    errors=[]
    for i in range(len(x_test)):
        if len(x_test[i])!=size:
            errors.append(i)
    for elem in sorted(errors, reverse=True):
        del x_test[elem]
        del y_test[elem]

    x_train, y_train = shuffle (x_train, y_train)
    x_val, y_val = shuffle(x_val, y_val)
    x_test, y_test = shuffle(x_test, y_test)
    return x_train, y_train, x_val, y_val, x_test, y_test

def gen_labels4(num, size):
    summarydata=artefact_summary()
    summarydata = summarydata.sample(1700) #random sampling
    train=summarydata[0:int(0.6*len(summarydata))]
    val=summarydata[int(0.6*len(summarydata)):int(0.8*len(summarydata))]
    test=summarydata[int(0.8*len(summarydata)):len(summarydata)]
    x_train=[]
    y_train=[]
    x_val=[]
    y_val=[]
    x_test=[]
    y_test=[]

    #normal
    for index, row in train.iterrows():
        device = str(row["deviceID"])
        start = str(row["interval_start"])
        end = str(row["artifact_finish"])
        url = str(genURL(device, start, end))
        df=genDF(url)

        df = df.drop(columns="Time")
        df = df.values.tolist()
        df = [val for sublist in df for val in sublist]
        data=np.array(df)
        scaler = MinMaxScaler()
        res = data.reshape(-1, 1)
        scaled = scaler.fit_transform(res)
        scaled = scaled.reshape(1, -1)
        df = scaled[0].tolist()

        rand_norm=rand_windows2(df[0:75000], num, size)
        rand_artifact=rand_windows2(df[75000:len(df)], num, size)
        for lis in rand_norm:
            x_train.append(lis)
            y_train.append(0)
        for lis in rand_artifact:
            x_train.append(lis)
            y_train.append(1)
    for index, row in val.iterrows():
        device = str(row["deviceID"])
        start = str(row["interval_start"])
        end = str(row["artifact_finish"])
        url = str(genURL(device, start, end))
        df=genDF(url)

        df = df.drop(columns="Time")
        df = df.values.tolist()
        df = [val for sublist in df for val in sublist]
        data=np.array(df)
        scaler = MinMaxScaler()
        res = data.reshape(-1, 1)
        scaled = scaler.fit_transform(res)
        scaled = scaled.reshape(1, -1)
        df = scaled[0].tolist()

        rand_norm=rand_windows2(df[0:75000], num, size)
        rand_artifact=rand_windows2(df[75000:len(df)], num, size)
        for lis in rand_norm:
            x_val.append(lis)
            y_val.append(0)
        for lis in rand_artifact:
            x_val.append(lis)
            y_val.append(1)
    for index, row in test.iterrows():
        device = str(row["deviceID"])
        start = str(row["interval_start"])
        end = str(row["artifact_finish"])
        url = str(genURL(device, start, end))
        df=genDF(url)

        df = df.drop(columns="Time")
        df = df.values.tolist()
        df = [val for sublist in df for val in sublist]
        data=np.array(df)
        scaler = MinMaxScaler()
        res = data.reshape(-1, 1)
        scaled = scaler.fit_transform(res)
        scaled = scaled.reshape(1, -1)
        df = scaled[0].tolist()

        rand_norm=rand_windows2(df[0:75000], num, size)
        rand_artifact=rand_windows2(df[75000:len(df)], num, size)
        for lis in rand_norm:
            x_test.append(lis)
            y_test.append(0)
        for lis in rand_artifact:
            x_test.append(lis)
            y_test.append(1)

    errors=[]
    for i in range(len(x_train)):
        if len(x_train[i])!=size:
            errors.append(i)
    for elem in sorted(errors, reverse=True):
        del x_train[elem]
        del y_train[elem]
    errors=[]
    for i in range(len(x_val)):
        if len(x_val[i])!=size:
            errors.append(i)
    for elem in sorted(errors, reverse=True):
        del x_val[elem]
        del y_val[elem]
    errors=[]
    for i in range(len(x_test)):
        if len(x_test[i])!=size:
            errors.append(i)
    for elem in sorted(errors, reverse=True):
        del x_test[elem]
        del y_test[elem]

    x_train, y_train = shuffle (x_train, y_train)
    x_val, y_val = shuffle(x_val, y_val)
    x_test, y_test = shuffle(x_test, y_test)
    return x_train, y_train, x_val, y_val, x_test, y_test

def gen_labels_sh():
    summarydata=artefact_summary()
    summarydata = summarydata.sample(1700) #random sampling
    #summarydata=summarydata[0:1700]
    size=7500
    train=summarydata[0:int(0.6*len(summarydata))]
    val=summarydata[int(0.6*len(summarydata)):int(0.8*len(summarydata))]
    test=summarydata[int(0.8*len(summarydata)):len(summarydata)]
    x_train=[]
    y_train=[]
    x_val=[]
    y_val=[]
    x_test=[]
    y_test=[]

    #normal
    for index, row in train.iterrows():
        device = str(row["deviceID"])
        start = str(int(row["interval_start"]))
        end = str(int(row["interval_start"])+600)
        url = str(genURL(device, start, end))
        df=genDF(url)

        df = df.drop(columns="Time")
        df = df.values.tolist()
        df = [val for sublist in df for val in sublist]
        x_train.append(df[0:size])
        y_train.append(0)
    for index, row in val.iterrows():
        device = str(row["deviceID"])
        start = str(int(row["interval_start"]))
        end = str(int(row["interval_start"])+600)
        url = str(genURL(device, start, end))
        df=genDF(url)

        df = df.drop(columns="Time")
        df = df.values.tolist()
        df = [val for sublist in df for val in sublist]
        x_val.append(df[0:size])
        y_val.append(0)
    for index, row in test.iterrows():
        device = str(row["deviceID"])
        start = str(int(row["interval_start"]))
        end = str(int(row["interval_start"])+600)
        url = str(genURL(device, start, end))
        df=genDF(url)

        df = df.drop(columns="Time")
        df = df.values.tolist()
        df = [val for sublist in df for val in sublist]
        x_test.append(df[0:size])
        y_test.append(0)

    print("here")
    #artefact
    for index, row in train.iterrows():
        device = str(row["deviceID"])
        start = str(int(row["artifact_start"]))
        end = str(int(row["artifact_start"])+600)
        url = str(genURL(device, start, end))
        artifactDF=genDF(url)

        artifactDF = artifactDF.drop(columns="Time")
        artifactDF = artifactDF.values.tolist()
        artifactDF = [val for sublist in artifactDF for val in sublist]
        x_train.append(artifactDF[0:size])
        y_train.append(1)
    for index, row in val.iterrows():
        device = str(row["deviceID"])
        start = str(int(row["artifact_start"]))
        end = str(int(row["artifact_start"])+600)
        url = str(genURL(device, start, end))
        artifactDF=genDF(url)

        artifactDF = artifactDF.drop(columns="Time")
        artifactDF = artifactDF.values.tolist()
        artifactDF = [val for sublist in artifactDF for val in sublist]
        x_val.append(artifactDF[0:size])
        y_val.append(1)
    for index, row in test.iterrows():
        device = str(row["deviceID"])
        start = str(int(row["artifact_start"]))
        end = str(int(row["artifact_start"])+600)
        url = str(genURL(device, start, end))
        artifactDF=genDF(url)

        artifactDF = artifactDF.drop(columns="Time")
        artifactDF = artifactDF.values.tolist()
        artifactDF = [val for sublist in artifactDF for val in sublist]
        x_test.append(artifactDF[0:size])
        y_test.append(1)

    errors=[]
    for i in range(len(x_train)):
        if len(x_train[i])!=size:
            errors.append(i)
    for elem in sorted(errors, reverse=True):
        del x_train[elem]
        del y_train[elem]
    errors=[]
    for i in range(len(x_val)):
        if len(x_val[i])!=size:
            errors.append(i)
    for elem in sorted(errors, reverse=True):
        del x_val[elem]
        del y_val[elem]
    errors=[]
    for i in range(len(x_test)):
        if len(x_test[i])!=size:
            errors.append(i)
    for elem in sorted(errors, reverse=True):
        del x_test[elem]
        del y_test[elem]

    x_train, y_train = shuffle (x_train, y_train)
    x_val, y_val = shuffle(x_val, y_val)
    x_test, y_test = shuffle(x_test, y_test)
    return x_train, y_train, x_val, y_val, x_test, y_test

def gen_labels_sh2():
    summarydata=artefact_summary()
    summarydata = summarydata.sample(1700) #random sampling
    #summarydata=summarydata[0:1700]
    size=7500
    train=summarydata[0:int(0.6*len(summarydata))]
    val=summarydata[int(0.6*len(summarydata)):int(0.8*len(summarydata))]
    test=summarydata[int(0.8*len(summarydata)):len(summarydata)]
    x_train=[]
    y_train=[]
    x_val=[]
    y_val=[]
    x_test=[]
    y_test=[]

    #normal
    for index, row in train.iterrows():
        device = str(row["deviceID"])
        start = str(int(row["interval_start"]))
        end = str(int(row["interval_start"])+600)
        url = str(genURL(device, start, end))
        df=genDF(url)

        df = df.drop(columns="Time")
        df = df.values.tolist()
        df = [val for sublist in df for val in sublist]
        x_train.append(df[0:size])
        y_train.append(0)
    for index, row in val.iterrows():
        device = str(row["deviceID"])
        start = str(int(row["interval_start"]))
        end = str(int(row["interval_start"])+600)
        url = str(genURL(device, start, end))
        df=genDF(url)

        df = df.drop(columns="Time")
        df = df.values.tolist()
        df = [val for sublist in df for val in sublist]
        x_val.append(df[0:size])
        y_val.append(0)
    for index, row in test.iterrows():
        device = str(row["deviceID"])
        start = str(int(row["interval_start"]))
        end = str(int(row["interval_start"])+600)
        url = str(genURL(device, start, end))
        df=genDF(url)

        df = df.drop(columns="Time")
        df = df.values.tolist()
        df = [val for sublist in df for val in sublist]
        x_test.append(df[0:size])
        y_test.append(0)

    print("here")
    #artefact
    for index, row in train.iterrows():
        device = str(row["deviceID"])
        start = str(int(row["artifact_start"]))
        end = str(int(row["artifact_start"])+600)
        url = str(genURL(device, start, end))
        artifactDF=genDF(url)

        artifactDF = artifactDF.drop(columns="Time")
        artifactDF = artifactDF.values.tolist()
        artifactDF = [val for sublist in artifactDF for val in sublist]
        x_train.append(artifactDF[0:size])
        y_train.append(1)
    for index, row in val.iterrows():
        device = str(row["deviceID"])
        start = str(int(row["artifact_start"]))
        end = str(int(row["artifact_start"])+600)
        url = str(genURL(device, start, end))
        artifactDF=genDF(url)

        artifactDF = artifactDF.drop(columns="Time")
        artifactDF = artifactDF.values.tolist()
        artifactDF = [val for sublist in artifactDF for val in sublist]
        x_val.append(artifactDF[0:size])
        y_val.append(1)
    for index, row in test.iterrows():
        device = str(row["deviceID"])
        start = str(int(row["artifact_start"]))
        end = str(int(row["artifact_start"])+600)
        url = str(genURL(device, start, end))
        artifactDF=genDF(url)

        artifactDF = artifactDF.drop(columns="Time")
        artifactDF = artifactDF.values.tolist()
        artifactDF = [val for sublist in artifactDF for val in sublist]
        x_test.append(artifactDF[0:size])
        y_test.append(1)

    errors=[]
    for i in range(len(x_train)):
        if len(x_train[i])!=size:
            errors.append(i)
    for elem in sorted(errors, reverse=True):
        del x_train[elem]
        del y_train[elem]
    errors=[]
    for i in range(len(x_val)):
        if len(x_val[i])!=size:
            errors.append(i)
    for elem in sorted(errors, reverse=True):
        del x_val[elem]
        del y_val[elem]
    errors=[]
    for i in range(len(x_test)):
        if len(x_test[i])!=size:
            errors.append(i)
    for elem in sorted(errors, reverse=True):
        del x_test[elem]
        del y_test[elem]

    x_train, y_train = shuffle (x_train, y_train)
    x_val, y_val = shuffle(x_val, y_val)
    x_test, y_test = shuffle(x_test, y_test)
    return x_train, y_train, x_val, y_val, x_test, y_test

def all_features2():
    ''' Returns dictionary of all features in the module

    .. note:: Some of the features (hist4, corr) are relatively expensive to compute
    '''
    features = {'mean': mean,
                'median': median,
                #'gmean': gmean,
                #'hmean': hmean,
                'vec_sum': vec_sum,
                #'abs_sum': abs_sum,
                'abs_energy': abs_energy,
                'std': std,
                'var': var,
                #'variation': variation,
                'min': minimum,
                'max': maximum,
                'skew': skew,
                'kurt': kurt,
                'mean_diff': mean_diff,
                'mean_abs_diff': means_abs_diff,
                'mse': mse,
                #'mnx': mean_crossings,
                #'hist4': hist(),
                #'corr': corr2,
                #'mean_abs_value': mean_abs,
                #'zero_crossings': zero_crossing(),
                'slope_sign_changes': slope_sign_changes(),
                'waveform_length': waveform_length,
                'emg_var': emg_var,
                #'root_mean_square': root_mean_square,
                #'willison_amplitude': willison_amplitude()
    }
    return features

def features_select():
    ''' Returns dictionary of all features in the module

    .. note:: Some of the features (hist4, corr) are relatively expensive to compute
    '''
    features = {#'mean': mean,
                #'median': median,
                #'gmean': gmean,
                #'vec_sum': vec_sum,
                #'abs_sum': abs_sum,
                #'abs_energy': abs_energy,
                'std': std,
                #'var': var,
                'min': minimum,
                'max': maximum,
                'skew': skew,
                'kurt': kurt,
                'mean_diff': mean_diff,
                'mean_abs_diff': means_abs_diff,
                #'mse': mse,
                'slope_sign_changes': slope_sign_changes(),
                #'waveform_length': waveform_length,
                #'emg_var': emg_var,
                #'root_mean_square': root_mean_square,
    }
    return features

def dwt_features(lis):
    out=[]
    for elem in lis:
        (ca, cd) = pywt.dwt(elem, 'db1')
        out.append(cd)
    return np.asarray(out)

def minmax():
    summarydata = artefact_summary()
    #summarydata = summarydata.sample(1700)  # random sampling

    # normal
    maxs=[]
    mins=[]
    for index, row in summarydata.iterrows():
        print(str(row["interval_ID"]))
        device = str(row["deviceID"])
        start = str(row["interval_start"])
        end = str(row["interval_finish"])
        url = str(genURL(device, start, end))
        df = genDF(url)
        df = df.drop(columns="Time")
        df = df.values.tolist()
        maxs.append(max(df))
        mins.append(min(df))
    return max(maxs),min(mins)

def genFigure():
    summarydata = pd.read_csv('artefact_summary.csv')
    summarydata = summarydata.sample(1)

    for index, row in summarydata.iterrows():
        url = row["URL"] + "&format=csv"
        df = genDF(url)

    adjustTime=df["Time"].iloc[0]
    df["Time"]=df["Time"].apply(lambda x: x-adjustTime)
    ax=df[0:625].plot("Time", "Pressure", legend=None, title="Arterial Pressure (mmHg) by Time (s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Arterial Pressure (mmHg)")

    ax1=df.plot("Time", "Pressure", legend=None, title="Arterial Pressure (mmHg) by Time (s)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Arterial Pressure (mmHg)")

    return df

def genShapelets(n):
    summarydata = artefact_summary()
    summarydata = summarydata.sample(n)  # random sampling

    shapelets=[]

    for index, row in summarydata.iterrows():
        device = str(row["deviceID"])
        start = str(row["interval_start"])
        end = str(row["artifact_finish"])
        url = str(genURL(device, start, end))
        df = genDF(url)

        df = df.drop(columns="Time")
        df = df.values.tolist()
        df = [val for sublist in df for val in sublist]
        data = np.array(df)
        scaler = StandardScaler()
        res = data.reshape(-1, 1)
        scaler.fit(res[0:75000])
        scaled = scaler.transform(res[75000:])
        scaled = scaled.reshape(1, -1)
        df = scaled[0].tolist()
        shapelets.append(df)
    return shapelets


#Generate labelled dataset with Standard Scaler fitted to normal data only
def gen_labels5(num, size):
    summarydata=artefact_summary()
    summarydata = summarydata.sample(1700) #random sampling
    train=summarydata[0:int(0.6*len(summarydata))]
    val=summarydata[int(0.6*len(summarydata)):int(0.8*len(summarydata))]
    test=summarydata[int(0.8*len(summarydata)):len(summarydata)]
    x_train=[]
    y_train=[]
    x_val=[]
    y_val=[]
    x_test=[]
    y_test=[]

    for index, row in train.iterrows():
        device = str(row["deviceID"])
        start = str(row["interval_start"])
        end = str(row["artifact_finish"])
        url = str(genURL(device, start, end))
        df=genDF(url)

        df = df.drop(columns="Time")
        df = df.values.tolist()
        df = [val for sublist in df for val in sublist]
        data=np.array(df)
        scaler = StandardScaler()
        res = data.reshape(-1, 1)
        scaler.fit(res[0:75000])
        scaled=scaler.transform(res)
        scaled = scaled.reshape(1, -1)
        df = scaled[0].tolist()

        rand_norm=rand_windows2(df[0:75000], num, size)
        rand_artifact=rand_windows2(df[75000:len(df)], num, size)
        for lis in rand_norm:
            x_train.append(lis)
            y_train.append(0)
        for lis in rand_artifact:
            x_train.append(lis)
            y_train.append(1)
    for index, row in val.iterrows():
        device = str(row["deviceID"])
        start = str(row["interval_start"])
        end = str(row["artifact_finish"])
        url = str(genURL(device, start, end))
        df=genDF(url)

        df = df.drop(columns="Time")
        df = df.values.tolist()
        df = [val for sublist in df for val in sublist]
        data=np.array(df)
        scaler = StandardScaler()
        res = data.reshape(-1, 1)
        scaler.fit(res[0:75000])
        scaled=scaler.transform(res)
        scaled = scaled.reshape(1, -1)
        df = scaled[0].tolist()

        rand_norm=rand_windows2(df[0:75000], num, size)
        rand_artifact=rand_windows2(df[75000:len(df)], num, size)
        for lis in rand_norm:
            x_val.append(lis)
            y_val.append(0)
        for lis in rand_artifact:
            x_val.append(lis)
            y_val.append(1)
    for index, row in test.iterrows():
        device = str(row["deviceID"])
        start = str(row["interval_start"])
        end = str(row["artifact_finish"])
        url = str(genURL(device, start, end))
        df=genDF(url)

        df = df.drop(columns="Time")
        df = df.values.tolist()
        df = [val for sublist in df for val in sublist]
        data=np.array(df)
        scaler = StandardScaler()
        res = data.reshape(-1, 1)
        scaler.fit(res[0:75000])
        scaled=scaler.transform(res)
        scaled = scaled.reshape(1, -1)
        df = scaled[0].tolist()

        rand_norm=rand_windows2(df[0:75000], num, size)
        rand_artifact=rand_windows2(df[75000:len(df)], num, size)
        for lis in rand_norm:
            x_test.append(lis)
            y_test.append(0)
        for lis in rand_artifact:
            x_test.append(lis)
            y_test.append(1)

    errors=[]
    for i in range(len(x_train)):
        if len(x_train[i])!=size:
            errors.append(i)
    for elem in sorted(errors, reverse=True):
        del x_train[elem]
        del y_train[elem]
    errors=[]
    for i in range(len(x_val)):
        if len(x_val[i])!=size:
            errors.append(i)
    for elem in sorted(errors, reverse=True):
        del x_val[elem]
        del y_val[elem]
    errors=[]
    for i in range(len(x_test)):
        if len(x_test[i])!=size:
            errors.append(i)
    for elem in sorted(errors, reverse=True):
        del x_test[elem]
        del y_test[elem]

    x_train, y_train = shuffle (x_train, y_train)
    x_val, y_val = shuffle(x_val, y_val)
    x_test, y_test = shuffle(x_test, y_test)
    return x_train, y_train, x_val, y_val, x_test, y_test

#Generate labelled dataset with Standard Scaler fitted to normal data only, and artefact/normal are scaled separately
def gen_labels6(num, size):
    summarydata=artefact_summary()
    summarydata = summarydata.sample(1700) #random sampling
    train=summarydata[0:int(0.6*len(summarydata))]
    val=summarydata[int(0.6*len(summarydata)):int(0.8*len(summarydata))]
    test=summarydata[int(0.8*len(summarydata)):len(summarydata)]
    x_train=[]
    y_train=[]
    x_val=[]
    y_val=[]
    x_test=[]
    y_test=[]

    for index, row in train.iterrows():
        device = str(row["deviceID"])
        start = str(row["interval_start"])
        end = str(row["artifact_finish"])
        url = str(genURL(device, start, end))
        df=genDF(url)

        df = df.drop(columns="Time")
        df = df.values.tolist()
        df = [val for sublist in df for val in sublist]
        data=np.array(df)
        if len(data)<(75000+size):
            print("here")
            continue

        scaler = StandardScaler()
        res = data.reshape(-1, 1)
        scaler.fit(res[0:75000])
        data = res.reshape(1, -1)
        df = data[0].tolist()

        rand_norm=rand_windows2(df[0:75000], num, size)
        rand_artifact=rand_windows2(df[75000:len(df)], num, size)
        for lis in rand_norm:
            lis = np.array(lis)
            res=lis.reshape(-1, 1)
            scaled=scaler.transform(res)
            scaled = scaled.reshape(1, -1)
            final = scaled[0].tolist()
            x_train.append(final)
            y_train.append(0)
        for lis in rand_artifact:
            lis = np.array(lis)
            res=lis.reshape(-1, 1)
            scaled=scaler.transform(res)
            scaled = scaled.reshape(1, -1)
            final = scaled[0].tolist()
            x_train.append(final)
            y_train.append(1)
    for index, row in val.iterrows():
        device = str(row["deviceID"])
        start = str(row["interval_start"])
        end = str(row["artifact_finish"])
        url = str(genURL(device, start, end))
        df=genDF(url)

        df = df.drop(columns="Time")
        df = df.values.tolist()
        df = [val for sublist in df for val in sublist]
        data=np.array(df)
        if len(data)<(75000+size):
            print("here")
            continue

        scaler = StandardScaler()
        res = data.reshape(-1, 1)
        scaler.fit(res[0:75000])
        data = res.reshape(1, -1)
        df = data[0].tolist()

        rand_norm = rand_windows2(df[0:75000], num, size)
        rand_artifact = rand_windows2(df[75000:len(df)], num, size)
        for lis in rand_norm:
            lis = np.array(lis)
            res = lis.reshape(-1, 1)
            scaled = scaler.transform(res)
            scaled = scaled.reshape(1, -1)
            final = scaled[0].tolist()
            x_val.append(final)
            y_val.append(0)
        for lis in rand_artifact:
            lis = np.array(lis)
            res = lis.reshape(-1, 1)
            scaled = scaler.transform(res)
            scaled = scaled.reshape(1, -1)
            final = scaled[0].tolist()
            x_val.append(final)
            y_val.append(1)
    for index, row in test.iterrows():
        device = str(row["deviceID"])
        start = str(row["interval_start"])
        end = str(row["artifact_finish"])
        url = str(genURL(device, start, end))
        df=genDF(url)

        df = df.drop(columns="Time")
        df = df.values.tolist()
        df = [val for sublist in df for val in sublist]
        data=np.array(df)
        if len(data)<(75000+size):
            print("here")
            continue

        scaler = StandardScaler()
        res = data.reshape(-1, 1)
        scaler.fit(res[0:75000])
        data = res.reshape(1, -1)
        df = data[0].tolist()

        rand_norm = rand_windows2(df[0:75000], num, size)
        rand_artifact = rand_windows2(df[75000:len(df)], num, size)
        for lis in rand_norm:
            lis=np.array(lis)
            res = lis.reshape(-1, 1)
            scaled = scaler.transform(res)
            scaled = scaled.reshape(1, -1)
            final = scaled[0].tolist()
            x_test.append(final)
            y_test.append(0)
        for lis in rand_artifact:
            lis = np.array(lis)
            res = lis.reshape(-1, 1)
            scaled = scaler.transform(res)
            scaled = scaled.reshape(1, -1)
            final = scaled[0].tolist()
            x_test.append(final)
            y_test.append(1)

    errors=[]
    for i in range(len(x_train)):
        if len(x_train[i])!=size:
            errors.append(i)
    for elem in sorted(errors, reverse=True):
        del x_train[elem]
        del y_train[elem]
    errors=[]
    for i in range(len(x_val)):
        if len(x_val[i])!=size:
            errors.append(i)
    for elem in sorted(errors, reverse=True):
        del x_val[elem]
        del y_val[elem]
    errors=[]
    for i in range(len(x_test)):
        if len(x_test[i])!=size:
            errors.append(i)
    for elem in sorted(errors, reverse=True):
        del x_test[elem]
        del y_test[elem]

    x_train, y_train = shuffle (x_train, y_train)
    x_val, y_val = shuffle(x_val, y_val)
    x_test, y_test = shuffle(x_test, y_test)
    return x_train, y_train, x_val, y_val, x_test, y_test

#Generate labelled dataset with Standard Scaler fitted to normal data only, and artefact/normal are scaled separately
#keeping as much non-artefact waveform as possible
def gen_labels7(size, num):
    summarydata=artefact_summary()
    summarydata = summarydata.sample(1700) #random sampling
    train=summarydata[0:int(0.6*len(summarydata))]
    val=summarydata[int(0.6*len(summarydata)):int(0.8*len(summarydata))]
    test=summarydata[int(0.8*len(summarydata)):len(summarydata)]
    x_train=[]
    y_train=[]
    x_val=[]
    y_val=[]
    x_test=[]
    y_test=[]


    for index, row in train.iterrows():
        device = str(row["deviceID"])
        start = str(row["interval_start"])
        end = str(row["artifact_finish"])
        url = str(genURL(device, start, end))
        df=genDF(url)

        df = df.drop(columns="Time")
        df = df.values.tolist()
        df = [val for sublist in df for val in sublist]
        data=np.array(df)

        device = str(row["deviceID"])
        start = str(row["artifact_finish"])
        end = str(row["interval_finish"])
        url = str(genURL(device, start, end))
        df1=genDF(url)

        df1 = df1.drop(columns="Time")
        df1 = df1.values.tolist()
        df1 = [val for sublist in df1 for val in sublist]
        data1=np.array(df1)

        combined = np.concatenate((data,data1))
        scaler = StandardScaler()
        res = combined.reshape(-1, 1)
        scaler.fit(res)

        df = data.tolist()
        df1=data1.tolist()

        rand_norm=rand_windows2(df[0:75000], num, size)
        rand_artifact=rand_windows2(df[75000:len(df)], num, size)
        rand_norm2=rand_windows2(df1, num, size)

        for lis in rand_norm:
            if len(lis)<size:
                continue
            lis = np.array(lis)
            res=lis.reshape(-1, 1)
            scaled=scaler.transform(res)
            scaled = scaled.reshape(1, -1)
            final = scaled[0].tolist()
            x_train.append(final)
            y_train.append(0)

        for lis in rand_norm2:
            if len(lis)<size:
                continue
            lis = np.array(lis)
            res=lis.reshape(-1, 1)
            scaled=scaler.transform(res)
            scaled = scaled.reshape(1, -1)
            final = scaled[0].tolist()
            x_train.append(final)
            y_train.append(0)

        for lis in rand_artifact:
            if len(lis)<size:
                continue
            lis = np.array(lis)
            res=lis.reshape(-1, 1)
            scaled=scaler.transform(res)
            scaled = scaled.reshape(1, -1)
            final = scaled[0].tolist()
            x_train.append(final)
            y_train.append(1)

    print("done train")
    for index, row in val.iterrows():
        device = str(row["deviceID"])
        start = str(row["interval_start"])
        end = str(row["artifact_finish"])
        url = str(genURL(device, start, end))
        df = genDF(url)

        df = df.drop(columns="Time")
        df = df.values.tolist()
        df = [val for sublist in df for val in sublist]
        data = np.array(df)

        device = str(row["deviceID"])
        start = str(row["artifact_finish"])
        end = str(row["interval_finish"])
        url = str(genURL(device, start, end))
        df1 = genDF(url)

        df1 = df1.drop(columns="Time")
        df1 = df1.values.tolist()
        df1 = [val for sublist in df1 for val in sublist]
        data1 = np.array(df1)

        combined = np.concatenate((data, data1))
        scaler = StandardScaler()
        res = combined.reshape(-1, 1)
        scaler.fit(res)

        df = data.tolist()
        df1 = data1.tolist()

        rand_norm = rand_windows2(df[0:75000], num, size)
        rand_artifact = rand_windows2(df[75000:len(df)], num, size)
        rand_norm2 = rand_windows2(df1, num, size)

        for lis in rand_norm:
            if len(lis)<size:
                continue
            lis = np.array(lis)
            res = lis.reshape(-1, 1)
            scaled = scaler.transform(res)
            scaled = scaled.reshape(1, -1)
            final = scaled[0].tolist()
            x_val.append(final)
            y_val.append(0)

        for lis in rand_norm2:
            if len(lis)<size:
                continue
            lis = np.array(lis)
            res = lis.reshape(-1, 1)
            scaled = scaler.transform(res)
            scaled = scaled.reshape(1, -1)
            final = scaled[0].tolist()
            x_val.append(final)
            y_val.append(0)

        for lis in rand_artifact:
            if len(lis)<size:
                continue
            lis = np.array(lis)
            res = lis.reshape(-1, 1)
            scaled = scaler.transform(res)
            scaled = scaled.reshape(1, -1)
            final = scaled[0].tolist()
            x_val.append(final)
            y_val.append(1)

    print("done val")
    for index, row in test.iterrows():
        device = str(row["deviceID"])
        start = str(row["interval_start"])
        end = str(row["artifact_finish"])
        url = str(genURL(device, start, end))
        df = genDF(url)

        df = df.drop(columns="Time")
        df = df.values.tolist()
        df = [val for sublist in df for val in sublist]
        data = np.array(df)

        device = str(row["deviceID"])
        start = str(row["artifact_finish"])
        end = str(row["interval_finish"])
        url = str(genURL(device, start, end))
        df1 = genDF(url)

        df1 = df1.drop(columns="Time")
        df1 = df1.values.tolist()
        df1 = [val for sublist in df1 for val in sublist]
        data1 = np.array(df1)

        combined = np.concatenate((data, data1))
        scaler = StandardScaler()
        res = combined.reshape(-1, 1)
        scaler.fit(res)

        df = data.tolist()
        df1 = data1.tolist()

        rand_norm = rand_windows2(df[0:75000], num, size)
        rand_artifact = rand_windows2(df[75000:len(df)], num, size)
        rand_norm2 = rand_windows2(df1, num, size)

        for lis in rand_norm:
            if len(lis)<size:
                continue
            lis = np.array(lis)
            res = lis.reshape(-1, 1)
            scaled = scaler.transform(res)
            scaled = scaled.reshape(1, -1)
            final = scaled[0].tolist()
            x_test.append(final)
            y_test.append(0)

        for lis in rand_norm2:
            if len(lis)<size:
                continue
            lis = np.array(lis)
            res = lis.reshape(-1, 1)
            scaled = scaler.transform(res)
            scaled = scaled.reshape(1, -1)
            final = scaled[0].tolist()
            x_test.append(final)
            y_test.append(0)

        for lis in rand_artifact:
            if len(lis)<size:
                continue
            lis = np.array(lis)
            res = lis.reshape(-1, 1)
            scaled = scaler.transform(res)
            scaled = scaled.reshape(1, -1)
            final = scaled[0].tolist()
            x_test.append(final)
            y_test.append(1)

    print("done test")
    errors=[]
    for i in range(len(x_train)):
        if len(x_train[i])!=size:
            errors.append(i)
    for elem in sorted(errors, reverse=True):
        del x_train[elem]
        del y_train[elem]
    errors=[]
    for i in range(len(x_val)):
        if len(x_val[i])!=size:
            errors.append(i)
    for elem in sorted(errors, reverse=True):
        del x_val[elem]
        del y_val[elem]
    errors=[]
    for i in range(len(x_test)):
        if len(x_test[i])!=size:
            errors.append(i)
    for elem in sorted(errors, reverse=True):
        del x_test[elem]
        del y_test[elem]

    x_train, y_train = shuffle (x_train, y_train)
    x_val, y_val = shuffle(x_val, y_val)
    x_test, y_test = shuffle(x_test, y_test)
    return x_train, y_train, x_val, y_val, x_test, y_test

#devices that have artefacts labelled
devices_artefact=['3-109_1', '3-110_1', '3-111_1', '3-75_1', '3-75_2']


def noise_labels():
    return pd.read_csv('output.csv')

def line_zero_labels():
    return pd.read_csv('labels.txt')

def dict_zero():
    dict = {'3-73_2':[]}
    pd=line_zero_labels()
    for index,row in pd.iterrows():
        device=str(row["deviceID"])
        start=row["start_time"]
        end=row["end_time"]
        tuple=(start,end)
        dict[device].append(tuple)
    return dict

def dict_noise():
    dict = {'3-73_2':[]}
    pd=noise_labels()
    for index,row in pd.iterrows():
        device=str(row["deviceID"])
        start=row["start"]
        end=row["finish"]
        tuple=(start,end)
        dict[device].append(tuple)
    return dict

def dict_artefacts():
    dict={'3-109_1':[], '3-111_1':[], '3-75_1':[], '3-75_2':[]}
    pd = artefact_summary()
    for index, row in pd.iterrows():
        if '3-110_1' != str(row['deviceID']):
            device=str(row['deviceID'])
            start=row['artifact_start']
            end=row['artifact_finish']
            tuple=(start,end)
            dict[device].append(tuple)
    seen=set()
    for key in dict.keys():
        dict[key] = sorted(dict[key])
        dict[key] = [(a,b) for a,b in dict[key]
                     if not (a in seen or seen.add(a))]
    for key in dict.keys():
        dict[key] = sorted(dict[key])
        dict[key] = [(a,b) for a,b in dict[key]
                     if not (b in seen or seen.add(b))]
    return dict


def gen_labels8(size, noise_num):
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []


    noise = dict_noise()
    dict_art = dict_artefacts()
    line = dict_zero()


    dict_art_train = {'3-109_1': [], '3-111_1': [], '3-75_1': [], '3-75_2': []}
    dict_art_val = {'3-109_1': [], '3-111_1': [], '3-75_1': [], '3-75_2': []}
    dict_art_test = {'3-109_1': [], '3-111_1': [], '3-75_1': [], '3-75_2': []}

    dict_noise_train = {'3-109_1': [], '3-110_1': [], '3-111_1': [], '3-75_1': [], '3-75_2': [], '3-73_2':[]}
    dict_noise_val = {'3-109_1': [], '3-110_1': [], '3-111_1': [], '3-75_1': [], '3-75_2': [], '3-73_2':[]}
    dict_noise_test = {'3-109_1': [], '3-110_1': [], '3-111_1': [], '3-75_1': [], '3-75_2': [], '3-73_2':[]}

    dict_line_train = {'3-109_1': [], '3-110_1': [], '3-111_1': [], '3-75_1': [], '3-75_2': [], '3-73_2':[]}
    dict_line_val = {'3-109_1': [], '3-110_1': [], '3-111_1': [], '3-75_1': [], '3-75_2': [], '3-73_2':[]}
    dict_line_test = {'3-109_1': [], '3-110_1': [], '3-111_1': [], '3-75_1': [], '3-75_2': [], '3-73_2':[]}


    #len(dict_art[key])
    for key in dict_art.keys():
        dict_art[key] = random.sample(dict_art[key], 10)
        dict_art_train[key] = dict_art[key][0:int(0.6 * len(dict_art[key]))]
        dict_art_val[key] = dict_art[key][int(0.6 * len(dict_art[key])):int(0.8 * len(dict_art[key]))]
        dict_art_test[key] = dict_art[key][int(0.8 * len(dict_art[key])):]


    for key in line.keys():
        if len(line[key])!=0:
            line[key] = random.sample(line[key], 10)
            dict_line_train[key] = line[key][0:int(0.6 * len(line[key]))]
            dict_line_val[key] = line[key][int(0.6 * len(line[key])):int(0.8 * len(line[key]))]
            dict_line_test[key] = line[key][int(0.8 * len(line[key])):]

    for key in noise.keys():
        if len(noise[key])!=0:
            noise[key] = random.sample(noise[key], noise_num)
            dict_noise_train[key] = noise[key][0:int(0.6 * len(noise[key]))]
            dict_noise_val[key] = noise[key][int(0.6 * len(noise[key])):int(0.8 * len(noise[key]))]
            dict_noise_test[key] = noise[key][int(0.8 * len(noise[key])):]

    timebefore = 10  # minutes

    for key in dict_noise_train.keys():
        for value in dict_noise_train[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)

            scaler = StandardScaler()
            reshape = data.reshape(-1, 1)  # shape that the scaler needs as an input
            scaler.fit(reshape)
            # get windows of the artefact region
            windows = rand_windows2(data[timebefore*60*125:], 10, size)
            # scale them
            for lis in windows:
                if len(lis) < size:
                    continue
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)
                # append to our dataset
                x_train.append(scaled[0])
                y_train.append(0)

    for key in dict_noise_val.keys():
        for value in dict_noise_val[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)

            scaler = StandardScaler()
            reshape = data.reshape(-1, 1)  # shape that the scaler needs as an input
            scaler.fit(reshape)
            # get windows of the artefact region
            windows = rand_windows2(data[timebefore*60*125:], 10, size)
            # scale them
            for lis in windows:
                if len(lis) < size:
                    continue
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)
                # append to our dataset
                x_val.append(scaled[0])
                y_val.append(0)

    for key in dict_noise_test.keys():
        for value in dict_noise_test[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)

            scaler = StandardScaler()
            reshape = data.reshape(-1, 1)  # shape that the scaler needs as an input
            scaler.fit(reshape)
            # get windows of the artefact region
            windows = rand_windows2(data[timebefore*60*125:], 10, size)
            # scale them
            for lis in windows:
                if len(lis) < size:
                    continue
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)
                # append to our dataset
                x_test.append(scaled[0])
                y_test.append(0)

####
    for key in dict_line_train.keys():
        for value in dict_line_train[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)

            scaler = StandardScaler()
            reshape = data.reshape(-1, 1)  # shape that the scaler needs as an input
            scaler.fit(reshape)
            # get windows of the artefact region
            windows = rand_windows2(data[timebefore*60*125:], 10, size)
            # scale them
            for lis in windows:
                if len(lis) < size:
                    continue
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)
                # append to our dataset
                x_train.append(scaled[0])
                y_train.append(0)

    for key in dict_line_val.keys():
        for value in dict_line_val[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)

            scaler = StandardScaler()
            reshape = data.reshape(-1, 1)  # shape that the scaler needs as an input
            scaler.fit(reshape)
            # get windows of the artefact region
            windows = rand_windows2(data[timebefore*60*125:], 10, size)
            # scale them
            for lis in windows:
                if len(lis) < size:
                    continue
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)
                # append to our dataset
                x_val.append(scaled[0])
                y_val.append(0)

    for key in dict_line_test.keys():
        for value in dict_line_test[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)

            scaler = StandardScaler()
            reshape = data.reshape(-1, 1)  # shape that the scaler needs as an input
            scaler.fit(reshape)
            # get windows of the artefact region
            windows = rand_windows2(data[timebefore*60*125:], 10, size)
            # scale them
            for lis in windows:
                if len(lis) < size:
                    continue
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)
                # append to our dataset
                x_test.append(scaled[0])
                y_test.append(0)


    for key in dict_art_train.keys():
        for value in dict_art_train[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)

            scaler = StandardScaler()
            reshape = data.reshape(-1, 1)  # shape that the scaler needs as an input
            scaler.fit(reshape)
            # get windows of the artefact region
            windows = rand_windows2(data[timebefore*60*125:], 10, size)
            # scale them
            for lis in windows:
                if len(lis) < size:
                    continue
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)
                # append to our dataset
                x_train.append(scaled[0])
                y_train.append(1)

    for key in dict_art_val.keys():
        for value in dict_art_val[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)

            scaler = StandardScaler()
            reshape = data.reshape(-1, 1)  # shape that the scaler needs as an input
            scaler.fit(reshape)
            # get windows of the artefact region
            windows = rand_windows2(data[timebefore*60*125:], 10, size)
            # scale them
            for lis in windows:
                if len(lis) < size:
                    continue
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)
                # append to our dataset
                x_val.append(scaled[0])
                y_val.append(1)
    for key in dict_art_test.keys():
        for value in dict_art_test[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)

            scaler = StandardScaler()
            reshape = data.reshape(-1, 1)  # shape that the scaler needs as an input
            scaler.fit(reshape)
            # get windows of the artefact region
            windows = rand_windows2(data[timebefore*60*125:], 10, size)
            # scale them
            for lis in windows:
                if len(lis) < size:
                    continue
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)
                # append to our dataset
                x_test.append(scaled[0])
                y_test.append(1)


    errors = []
    for i in range(len(x_train)):
        if len(x_train[i]) != size:
            errors.append(i)
    for elem in sorted(errors, reverse=True):
        del x_train[elem]
        del y_train[elem]
    errors = []
    for i in range(len(x_val)):
        if len(x_val[i]) != size:
            errors.append(i)
    for elem in sorted(errors, reverse=True):
        del x_val[elem]
        del y_val[elem]
    errors = []
    for i in range(len(x_test)):
        if len(x_test[i]) != size:
            errors.append(i)
    for elem in sorted(errors, reverse=True):
        del x_test[elem]
        del y_test[elem]

    x_train, y_train = shuffle(x_train, y_train)
    x_val, y_val = shuffle(x_val, y_val)
    x_test, y_test = shuffle(x_test, y_test)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return x_train, y_train, x_val, y_val, x_test, y_test


##only start of artefact
def gen_labels9(size):
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []


    noise = dict_noise()
    dict_art = dict_artefacts()
    line = dict_zero()


    dict_art_train = {'3-109_1': [], '3-111_1': [], '3-75_1': [], '3-75_2': []}
    dict_art_val = {'3-109_1': [], '3-111_1': [], '3-75_1': [], '3-75_2': []}
    dict_art_test = {'3-109_1': [], '3-111_1': [], '3-75_1': [], '3-75_2': []}

    dict_noise_train = {'3-73_2':[]}
    dict_noise_val = {'3-73_2':[]}
    dict_noise_test = {'3-73_2':[]}

    dict_line_train = {'3-73_2':[]}
    dict_line_val = {'3-73_2':[]}
    dict_line_test = { '3-73_2':[]}


    #len(dict_art[key])
    for key in dict_art.keys():
        dict_art[key] = random.sample(dict_art[key], len(dict_art[key]))
        dict_art_train[key] = dict_art[key][0:int(0.6 * len(dict_art[key]))]
        dict_art_val[key] = dict_art[key][int(0.6 * len(dict_art[key])):int(0.8 * len(dict_art[key]))]
        dict_art_test[key] = dict_art[key][int(0.8 * len(dict_art[key])):]


    for key in line.keys():
        if len(line[key])!=0:
            line[key] = random.sample(line[key], len(line[key]))
            dict_line_train[key] = line[key][0:int(0.6 * len(line[key]))]
            dict_line_val[key] = line[key][int(0.6 * len(line[key])):int(0.8 * len(line[key]))]
            dict_line_test[key] = line[key][int(0.8 * len(line[key])):]

    for key in noise.keys():
        if len(noise[key])!=0:
            noise[key] = random.sample(noise[key], len(noise[key]))
            dict_noise_train[key] = noise[key][0:int(0.6 * len(noise[key]))]
            dict_noise_val[key] = noise[key][int(0.6 * len(noise[key])):int(0.8 * len(noise[key]))]
            dict_noise_test[key] = noise[key][int(0.8 * len(noise[key])):]

    timebefore = 10  # minutes

    for key in dict_noise_train.keys():
        for value in dict_noise_train[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)



            windows = rand_windows3(data, len(data)//size, (timebefore * 60 * 125), size)
            # scale them
            for window in windows:
                if (window[1] - window[0]) < size or isinstance(window[1], float) or isinstance(window[0], float):
                    continue

                # lis
                lis = data[window[0]:window[1]]

                # 10 min before list
                before = data[window[0] - (timebefore * 60 * 125):window[0]]
                if len(before)!= (timebefore * 60 * 125):
                    continue
                # calculate scaler based on 10 min before
                scaler = StandardScaler()
                reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
                scaler.fit(reshape)

                # scale lis using scaler
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)

                # save
                # append to our dataset
                x_train.append(scaled[0])
                y_train.append(0)

    for key in dict_noise_val.keys():
        for value in dict_noise_val[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)


            # get window of the start of artefact region

            windows = rand_windows3(data, len(data)//size, (timebefore * 60 * 125), size)
            # scale them
            for window in windows:
                if (window[1] - window[0]) < size or isinstance(window[1], float) or isinstance(window[0], float):
                    continue

                # lis
                lis = data[window[0]:window[1]]

                # 10 min before list
                before = data[window[0] - (timebefore * 60 * 125):window[0]]
                if len(before)!= (timebefore * 60 * 125):
                    continue
                # calculate scaler based on 10 min before
                scaler = StandardScaler()
                reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
                scaler.fit(reshape)

                # scale lis using scaler
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)

                # save
                # append to our dataset
                x_val.append(scaled[0])
                y_val.append(0)

    for key in dict_noise_test.keys():
        for value in dict_noise_test[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)


            # get window of the start of artefact region

            windows = rand_windows3(data, len(data)//size, (timebefore * 60 * 125), size)
            # scale them
            for window in windows:
                if (window[1] - window[0]) < size or isinstance(window[1], float) or isinstance(window[0], float):
                    continue

                # lis
                lis = data[window[0]:window[1]]

                # 10 min before list
                before = data[window[0] - (timebefore * 60 * 125):window[0]]
                if len(before)!= (timebefore * 60 * 125):
                    continue
                # calculate scaler based on 10 min before
                scaler = StandardScaler()
                reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
                scaler.fit(reshape)

                # scale lis using scaler
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)

                # save
                # append to our dataset
                x_test.append(scaled[0])
                y_test.append(0)

####
    for key in dict_line_train.keys():
        for value in dict_line_train[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)


            # get window of the start of artefact region

            windows = rand_windows3(data, len(data)//size, (timebefore * 60 * 125), size)
            # scale them
            for window in windows:
                if (window[1] - window[0]) < size or isinstance(window[1], float) or isinstance(window[0], float):
                    continue

                # lis
                lis = data[window[0]:window[1]]

                # 10 min before list
                before = data[window[0] - (timebefore * 60 * 125):window[0]]
                if len(before)!= (timebefore * 60 * 125):
                    continue
                # calculate scaler based on 10 min before
                scaler = StandardScaler()
                reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
                scaler.fit(reshape)

                # scale lis using scaler
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)

                # save
                # append to our dataset
                x_train.append(scaled[0])
                y_train.append(0)

    for key in dict_line_val.keys():
        for value in dict_line_val[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)


            # get window of the start of artefact region

            windows = rand_windows3(data, len(data)//size, (timebefore * 60 * 125), size)
            # scale them
            for window in windows:
                if (window[1] - window[0]) < size or isinstance(window[1], float) or isinstance(window[0], float):
                    continue

                # lis
                lis = data[window[0]:window[1]]

                # 10 min before list
                before = data[window[0] - (timebefore * 60 * 125):window[0]]
                if len(before)!= (timebefore * 60 * 125):
                    continue
                # calculate scaler based on 10 min before
                scaler = StandardScaler()
                reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
                scaler.fit(reshape)

                # scale lis using scaler
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)

                # save
                # append to our dataset
                x_val.append(scaled[0])
                y_val.append(0)

    for key in dict_line_test.keys():
        for value in dict_line_test[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)


            # get window of the start of artefact region

            windows = rand_windows3(data, len(data)//size, (timebefore * 60 * 125), size)
            # scale them
            for window in windows:
                if (window[1] - window[0]) < size or isinstance(window[1], float) or isinstance(window[0], float):
                    continue

                # lis
                lis = data[window[0]:window[1]]

                # 10 min before list
                before = data[window[0] - (timebefore * 60 * 125):window[0]]
                if len(before)!= (timebefore * 60 * 125):
                    continue
                # calculate scaler based on 10 min before
                scaler = StandardScaler()
                reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
                scaler.fit(reshape)

                # scale lis using scaler
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)

                # save
                # append to our dataset
                x_test.append(scaled[0])
                y_test.append(0)


    for key in dict_art_train.keys():
        for value in dict_art_train[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1]+size//125)))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)

            if len(data)<timebefore*60*125:
                continue

            before = data[:timebefore*60*125]
            if len(before)!= (timebefore * 60 * 125):
                continue
            scaler = StandardScaler()
            reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
            scaler.fit(reshape)
            # get window of the start of artefact region

            window=data[timebefore*60*125:(timebefore * 60 * 125)+size]
            # scale them
            if len(window) < size:
                continue
            res = window.reshape(-1, 1)
            scaled = scaler.transform(res)
            scaled = scaled.reshape(1, -1)
            # append to our dataset
            x_train.append(scaled[0])
            y_train.append(1)

    for key in dict_art_val.keys():
        for value in dict_art_val[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1]+size//125)))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)

            if len(data)<timebefore*60*125:
                continue

            before = data[:timebefore*60*125]
            if len(before)!= (timebefore * 60 * 125):
                continue
            scaler = StandardScaler()
            reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
            scaler.fit(reshape)
            # get window of the start of artefact region

            window = data[timebefore * 60 * 125:(timebefore * 60 * 125)+size]
            # scale them
            if len(window) < size:
                continue
            res = window.reshape(-1, 1)
            scaled = scaler.transform(res)
            scaled = scaled.reshape(1, -1)
            # append to our dataset
            x_val.append(scaled[0])
            y_val.append(1)

    for key in dict_art_test.keys():
        for value in dict_art_test[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1]+size//125)))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)

            if len(data)<timebefore*60*125:
                continue

            before = data[:timebefore*60*125]
            if len(before)!= (timebefore * 60 * 125):
                continue
            scaler = StandardScaler()
            reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
            scaler.fit(reshape)
            # get window of the start of artefact region

            window = data[timebefore * 60 * 125:(timebefore * 60 * 125)+size]
            # scale them
            if len(window) < size:
                continue
            res = window.reshape(-1, 1)
            scaled = scaler.transform(res)
            scaled = scaled.reshape(1, -1)
            # append to our dataset
            x_test.append(scaled[0])
            y_test.append(1)


    errors = []
    for i in range(len(x_train)):
        if len(x_train[i]) != size:
            errors.append(i)
    for elem in sorted(errors, reverse=True):
        del x_train[elem]
        del y_train[elem]
    errors = []
    for i in range(len(x_val)):
        if len(x_val[i]) != size:
            errors.append(i)
    for elem in sorted(errors, reverse=True):
        del x_val[elem]
        del y_val[elem]
    errors = []
    for i in range(len(x_test)):
        if len(x_test[i]) != size:
            errors.append(i)
    for elem in sorted(errors, reverse=True):
        del x_test[elem]
        del y_test[elem]

    x_train, y_train = shuffle(x_train, y_train)
    x_val, y_val = shuffle(x_val, y_val)
    x_test, y_test = shuffle(x_test, y_test)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return x_train, y_train, x_val, y_val, x_test, y_test

#Proper scaling same was as streaming inference
#random sampling of windows in sharkfin region
def gen_labels10(size):
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []


    noise = dict_noise()
    dict_art = dict_artefacts()
    line = dict_zero()


    dict_art_train = {'3-109_1': [], '3-111_1': [], '3-75_1': [], '3-75_2': []}
    dict_art_val = {'3-109_1': [], '3-111_1': [], '3-75_1': [], '3-75_2': []}
    dict_art_test = {'3-109_1': [], '3-111_1': [], '3-75_1': [], '3-75_2': []}

    dict_noise_train = {'3-73_2':[]}
    dict_noise_val = {'3-73_2':[]}
    dict_noise_test = {'3-73_2':[]}

    dict_line_train = {'3-73_2':[]}
    dict_line_val = {'3-73_2':[]}
    dict_line_test = { '3-73_2':[]}


    #SPLITTING DATA AT THE LEVEL OF THE ARTEFACT, LINE ZERO, NOISE
    #So that the windows from the same artefact are not found in more than one of train, val, or test
    for key in dict_art.keys():
        dict_art[key] = random.sample(dict_art[key], len(dict_art[key]))
        dict_art_train[key] = dict_art[key][0:int(0.6 * len(dict_art[key]))]
        dict_art_val[key] = dict_art[key][int(0.6 * len(dict_art[key])):int(0.8 * len(dict_art[key]))]
        dict_art_test[key] = dict_art[key][int(0.8 * len(dict_art[key])):]


    for key in line.keys():
        if len(line[key])!=0:
            line[key] = random.sample(line[key], len(line[key]))
            dict_line_train[key] = line[key][0:int(0.6 * len(line[key]))]
            dict_line_val[key] = line[key][int(0.6 * len(line[key])):int(0.8 * len(line[key]))]
            dict_line_test[key] = line[key][int(0.8 * len(line[key])):]

    for key in noise.keys():
        if len(noise[key])!=0:
            noise[key] = random.sample(noise[key], len(noise[key]))
            dict_noise_train[key] = noise[key][0:int(0.6 * len(noise[key]))]
            dict_noise_val[key] = noise[key][int(0.6 * len(noise[key])):int(0.8 * len(noise[key]))]
            dict_noise_test[key] = noise[key][int(0.8 * len(noise[key])):]

    timebefore = 10  # minutes

    for key in tqdm(dict_noise_train.keys()):
        for value in dict_noise_train[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)

            # get window of the start of artefact region

            windows = rand_windows3(data, len(data)//size, (timebefore * 60 * 125), size)
            # scale them
            for window in windows:
                if (window[1] - window[0]) != size:
                    continue

                # lis
                lis = data[window[0]:window[1]]

                # 10 min before list
                before = data[window[0] - (timebefore * 60 * 125):window[0]]
                if len(before)!= (timebefore * 60 * 125):
                    continue
                # calculate scaler based on 10 min before
                scaler = StandardScaler()
                reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
                scaler.fit(reshape)

                # scale lis using scaler
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)

                # save
                # append to our dataset
                x_train.append(scaled[0])
                y_train.append(0)

    for key in tqdm(dict_noise_val.keys()):
        for value in dict_noise_val[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)

            # get window of the start of artefact region

            windows = rand_windows3(data, len(data)//size, (timebefore * 60 * 125), size)
            # scale them
            for window in windows:
                if (window[1] - window[0]) != size:
                    continue

                # lis
                lis = data[window[0]:window[1]]

                # 10 min before list
                before = data[window[0] - (timebefore * 60 * 125):window[0]]
                if len(before)!= (timebefore * 60 * 125):
                    continue
                # calculate scaler based on 10 min before
                scaler = StandardScaler()
                reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
                scaler.fit(reshape)

                # scale lis using scaler
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)

                # save
                # append to our dataset
                x_val.append(scaled[0])
                y_val.append(0)

    for key in tqdm(dict_noise_test.keys()):
        for value in dict_noise_test[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)

            # get window of the start of artefact region

            windows = rand_windows3(data, len(data)//size, (timebefore * 60 * 125), size)
            # scale them
            for window in windows:
                if (window[1] - window[0]) != size:
                    continue

                # lis
                lis = data[window[0]:window[1]]

                # 10 min before list
                before = data[window[0] - (timebefore * 60 * 125):window[0]]
                if len(before)!= (timebefore * 60 * 125):
                    continue
                # calculate scaler based on 10 min before
                scaler = StandardScaler()
                reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
                scaler.fit(reshape)

                # scale lis using scaler
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)

                # save
                # append to our dataset
                x_test.append(scaled[0])
                y_test.append(0)

####
    for key in tqdm(dict_line_train.keys()):
        for value in dict_line_train[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)

            # get window of the start of artefact region

            windows = rand_windows3(data, len(data)//size, (timebefore * 60 * 125), size)
            # scale them
            for window in windows:
                if (window[1] - window[0]) != size:
                    continue

                # lis
                lis = data[window[0]:window[1]]

                # 10 min before list
                before = data[window[0] - (timebefore * 60 * 125):window[0]]
                if len(before)!= (timebefore * 60 * 125):
                    continue
                # calculate scaler based on 10 min before
                scaler = StandardScaler()
                reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
                scaler.fit(reshape)

                # scale lis using scaler
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)

                # save
                # append to our dataset
                x_train.append(scaled[0])
                y_train.append(0)

    for key in tqdm(dict_line_val.keys()):
        for value in dict_line_val[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)

            # get window of the start of artefact region

            windows = rand_windows3(data, len(data)//size, (timebefore * 60 * 125), size)
            # scale them
            for window in windows:
                if (window[1] - window[0]) != size:
                    continue

                # lis
                lis = data[window[0]:window[1]]

                # 10 min before list
                before = data[window[0] - (timebefore * 60 * 125):window[0]]
                if len(before)!= (timebefore * 60 * 125):
                    continue
                # calculate scaler based on 10 min before
                scaler = StandardScaler()
                reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
                scaler.fit(reshape)

                # scale lis using scaler
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)

                # save
                # append to our dataset
                x_val.append(scaled[0])
                y_val.append(0)

    for key in tqdm(dict_line_test.keys()):
        for value in dict_line_test[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)

            # get window of the start of artefact region

            windows = rand_windows3(data, len(data)//size, (timebefore * 60 * 125), size)
            # scale them
            for window in windows:
                if (window[1] - window[0]) != size:
                    continue

                # lis
                lis = data[window[0]:window[1]]

                # 10 min before list
                before = data[window[0] - (timebefore * 60 * 125):window[0]]
                if len(before)!= (timebefore * 60 * 125):
                    continue
                # calculate scaler based on 10 min before
                scaler = StandardScaler()
                reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
                scaler.fit(reshape)

                # scale lis using scaler
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)

                # save
                # append to our dataset
                x_test.append(scaled[0])
                y_test.append(0)


    for key in tqdm(dict_art_train.keys()):
        for value in dict_art_train[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)

            # get window of the start of artefact region

            windows = rand_windows3(data, len(data)//size, (timebefore * 60 * 125), size)
            # scale them
            for window in windows:
                if (window[1]-window[0]) != size:
                    continue

                # lis
                lis = data[window[0]:window[1]]

                # 10 min before list
                before = data[window[0]-(timebefore*60*125):window[0]]
                if len(before)!= (timebefore * 60 * 125):
                    continue
                # calculate scaler based on 10 min before
                scaler = StandardScaler()
                reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
                scaler.fit(reshape)

                #scale lis using scaler
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)

                #save
                # append to our dataset
                x_train.append(scaled[0])
                y_train.append(1)

    for key in tqdm(dict_art_val.keys()):
        for value in dict_art_val[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)

            # get window of the start of artefact region

            windows = rand_windows3(data, len(data)//size, (timebefore * 60 * 125), size)
            # scale them
            for window in windows:
                if (window[1] - window[0]) != size:
                    continue

                # lis
                lis = data[window[0]:window[1]]

                # 10 min before list
                before = data[window[0] - (timebefore * 60 * 125):window[0]]
                if len(before)!= (timebefore * 60 * 125):
                    continue
                # calculate scaler based on 10 min before
                scaler = StandardScaler()
                reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
                scaler.fit(reshape)

                # scale lis using scaler
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)

                # save
                # append to our dataset
                x_val.append(scaled[0])
                y_val.append(1)

    for key in tqdm(dict_art_test.keys()):
        for value in dict_art_test[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)

            # get window of the start of artefact region

            windows = rand_windows3(data, len(data)//size, (timebefore * 60 * 125), size)
            # scale them
            for window in windows:
                if (window[1] - window[0]) != size:
                    continue

                # lis
                lis = data[window[0]:window[1]]

                # 10 min before list
                before = data[window[0] - (timebefore * 60 * 125):window[0]]
                if len(before)!= (timebefore * 60 * 125):
                    continue
                # calculate scaler based on 10 min before
                scaler = StandardScaler()
                reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
                scaler.fit(reshape)

                # scale lis using scaler
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)

                # save
                # append to our dataset
                x_test.append(scaled[0])
                y_test.append(1)


    errors = []
    for i in range(len(x_train)):
        if len(x_train[i]) != size:
            errors.append(i)
    for elem in sorted(errors, reverse=True):
        del x_train[elem]
        del y_train[elem]
    errors = []
    for i in range(len(x_val)):
        if len(x_val[i]) != size:
            errors.append(i)
    for elem in sorted(errors, reverse=True):
        del x_val[elem]
        del y_val[elem]
    errors = []
    for i in range(len(x_test)):
        if len(x_test[i]) != size:
            errors.append(i)
    for elem in sorted(errors, reverse=True):
        del x_test[elem]
        del y_test[elem]

    x_train, y_train = shuffle(x_train, y_train)
    x_val, y_val = shuffle(x_val, y_val)
    x_test, y_test = shuffle(x_test, y_test)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return x_train, y_train, x_val, y_val, x_test, y_test


##only start of artefact
##Random perturbation of start
def gen_labels11(size, shift):
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []


    noise = dict_noise()
    dict_art = dict_artefacts()
    line = dict_zero()


    dict_art_train = {'3-109_1': [], '3-111_1': [], '3-75_1': [], '3-75_2': []}
    dict_art_val = {'3-109_1': [], '3-111_1': [], '3-75_1': [], '3-75_2': []}
    dict_art_test = {'3-109_1': [], '3-111_1': [], '3-75_1': [], '3-75_2': []}

    dict_noise_train = {'3-73_2':[]}
    dict_noise_val = {'3-73_2':[]}
    dict_noise_test = {'3-73_2':[]}

    dict_line_train = {'3-73_2':[]}
    dict_line_val = {'3-73_2':[]}
    dict_line_test = { '3-73_2':[]}


    #len(dict_art[key])
    for key in dict_art.keys():
        dict_art[key] = random.sample(dict_art[key], len(dict_art[key]))
        dict_art_train[key] = dict_art[key][0:int(0.6 * len(dict_art[key]))]
        dict_art_val[key] = dict_art[key][int(0.6 * len(dict_art[key])):int(0.8 * len(dict_art[key]))]
        dict_art_test[key] = dict_art[key][int(0.8 * len(dict_art[key])):]


    for key in line.keys():
        if len(line[key])!=0:
            line[key] = random.sample(line[key], len(line[key]))
            dict_line_train[key] = line[key][0:int(0.6 * len(line[key]))]
            dict_line_val[key] = line[key][int(0.6 * len(line[key])):int(0.8 * len(line[key]))]
            dict_line_test[key] = line[key][int(0.8 * len(line[key])):]

    for key in noise.keys():
        if len(noise[key])!=0:
            noise[key] = random.sample(noise[key], len(noise[key]))
            dict_noise_train[key] = noise[key][0:int(0.6 * len(noise[key]))]
            dict_noise_val[key] = noise[key][int(0.6 * len(noise[key])):int(0.8 * len(noise[key]))]
            dict_noise_test[key] = noise[key][int(0.8 * len(noise[key])):]

    timebefore = 10  # minutes

    for key in dict_noise_train.keys():
        for value in dict_noise_train[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)



            windows = rand_windows3(data, len(data)//size, (timebefore * 60 * 125), size)
            # scale them
            for window in windows:
                if (window[1] - window[0]) < size:
                    continue

                # lis
                lis = data[window[0]:window[1]]

                # 10 min before list
                before = data[window[0] - (timebefore * 60 * 125):window[0]]
                if len(before)!= (timebefore * 60 * 125):
                    continue
                # calculate scaler based on 10 min before
                scaler = StandardScaler()
                reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
                scaler.fit(reshape)

                # scale lis using scaler
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)

                # save
                # append to our dataset
                x_train.append(scaled[0])
                y_train.append(0)

    for key in dict_noise_val.keys():
        for value in dict_noise_val[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)


            # get window of the start of artefact region

            windows = rand_windows3(data, len(data)//size, (timebefore * 60 * 125), size)
            # scale them
            for window in windows:
                if (window[1] - window[0]) < size:
                    continue

                # lis
                lis = data[window[0]:window[1]]

                # 10 min before list
                before = data[window[0] - (timebefore * 60 * 125):window[0]]
                if len(before)!= (timebefore * 60 * 125):
                    continue
                # calculate scaler based on 10 min before
                scaler = StandardScaler()
                reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
                scaler.fit(reshape)

                # scale lis using scaler
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)

                # save
                # append to our dataset
                x_val.append(scaled[0])
                y_val.append(0)

    for key in dict_noise_test.keys():
        for value in dict_noise_test[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)


            # get window of the start of artefact region

            windows = rand_windows3(data, len(data)//size, (timebefore * 60 * 125), size)
            # scale them
            for window in windows:
                if (window[1] - window[0]) < size:
                    continue

                # lis
                lis = data[window[0]:window[1]]

                # 10 min before list
                before = data[window[0] - (timebefore * 60 * 125):window[0]]
                if len(before)!= (timebefore * 60 * 125):
                    continue
                # calculate scaler based on 10 min before
                scaler = StandardScaler()
                reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
                scaler.fit(reshape)

                # scale lis using scaler
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)

                # save
                # append to our dataset
                x_test.append(scaled[0])
                y_test.append(0)

####
    for key in dict_line_train.keys():
        for value in dict_line_train[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)


            # get window of the start of artefact region

            windows = rand_windows3(data, len(data)//size, (timebefore * 60 * 125), size)
            # scale them
            for window in windows:
                if (window[1] - window[0]) < size:
                    continue

                # lis
                lis = data[window[0]:window[1]]

                # 10 min before list
                before = data[window[0] - (timebefore * 60 * 125):window[0]]
                if len(before)!= (timebefore * 60 * 125):
                    continue
                # calculate scaler based on 10 min before
                scaler = StandardScaler()
                reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
                scaler.fit(reshape)

                # scale lis using scaler
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)

                # save
                # append to our dataset
                x_train.append(scaled[0])
                y_train.append(0)

    for key in dict_line_val.keys():
        for value in dict_line_val[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)


            # get window of the start of artefact region

            windows = rand_windows3(data, len(data)//size, (timebefore * 60 * 125), size)
            # scale them
            for window in windows:
                if (window[1] - window[0]) < size:
                    continue

                # lis
                lis = data[window[0]:window[1]]

                # 10 min before list
                before = data[window[0] - (timebefore * 60 * 125):window[0]]
                if len(before)!= (timebefore * 60 * 125):
                    continue
                # calculate scaler based on 10 min before
                scaler = StandardScaler()
                reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
                scaler.fit(reshape)

                # scale lis using scaler
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)

                # save
                # append to our dataset
                x_val.append(scaled[0])
                y_val.append(0)

    for key in dict_line_test.keys():
        for value in dict_line_test[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)


            # get window of the start of artefact region

            windows = rand_windows3(data, len(data)//size, (timebefore * 60 * 125), size)
            # scale them
            for window in windows:
                if (window[1] - window[0]) < size:
                    continue

                # lis
                lis = data[window[0]:window[1]]

                # 10 min before list
                before = data[window[0] - (timebefore * 60 * 125):window[0]]
                if len(before)!= (timebefore * 60 * 125):
                    continue
                # calculate scaler based on 10 min before
                scaler = StandardScaler()
                reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
                scaler.fit(reshape)

                # scale lis using scaler
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)

                # save
                # append to our dataset
                x_test.append(scaled[0])
                y_test.append(0)


    for key in dict_art_train.keys():
        for value in dict_art_train[key]:
            shifted_start = value[0] + random.randint(-shift,0)
            url = str(genURL(str(key), str(shifted_start-(timebefore*60)), str(value[1]+size//125)))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)

            before = data[:timebefore*60*125]
            if len(before)!= (timebefore * 60 * 125):
                continue
            scaler = StandardScaler()
            reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
            scaler.fit(reshape)
            # get window of the start of artefact region

            window=data[timebefore*60*125:(timebefore * 60 * 125)+size]
            # scale them
            if len(window) < size:
                continue
            res = window.reshape(-1, 1)
            scaled = scaler.transform(res)
            scaled = scaled.reshape(1, -1)
            # append to our dataset
            x_train.append(scaled[0])
            y_train.append(1)

    for key in dict_art_val.keys():
        for value in dict_art_val[key]:
            shifted_start = value[0] + random.randint(-shift, 0)
            url = str(genURL(str(key), str(shifted_start - (timebefore * 60)), str(value[1] + size//125)))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)

            before = data[:timebefore*60*125]
            if len(before)!= (timebefore * 60 * 125):
                continue
            scaler = StandardScaler()
            reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
            scaler.fit(reshape)
            # get window of the start of artefact region

            window = data[timebefore * 60 * 125:(timebefore * 60 * 125)+size]
            # scale them
            if len(window) < size:
                continue
            res = window.reshape(-1, 1)
            scaled = scaler.transform(res)
            scaled = scaled.reshape(1, -1)
            # append to our dataset
            x_val.append(scaled[0])
            y_val.append(1)

    for key in dict_art_test.keys():
        for value in dict_art_test[key]:
            shifted_start = value[0] + random.randint(-shift, 0)
            url = str(genURL(str(key), str(shifted_start - (timebefore * 60)), str(value[1] + size//125)))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)

            before = data[:timebefore*60*125]
            if len(before)!= (timebefore * 60 * 125):
                continue
            scaler = StandardScaler()
            reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
            scaler.fit(reshape)
            # get window of the start of artefact region

            window = data[timebefore * 60 * 125:(timebefore * 60 * 125)+size]
            # scale them
            if len(window) < size:
                continue
            res = window.reshape(-1, 1)
            scaled = scaler.transform(res)
            scaled = scaled.reshape(1, -1)
            # append to our dataset
            x_test.append(scaled[0])
            y_test.append(1)


    errors = []
    for i in range(len(x_train)):
        if len(x_train[i]) != size:
            errors.append(i)
    for elem in sorted(errors, reverse=True):
        del x_train[elem]
        del y_train[elem]
    errors = []
    for i in range(len(x_val)):
        if len(x_val[i]) != size:
            errors.append(i)
    for elem in sorted(errors, reverse=True):
        del x_val[elem]
        del y_val[elem]
    errors = []
    for i in range(len(x_test)):
        if len(x_test[i]) != size:
            errors.append(i)
    for elem in sorted(errors, reverse=True):
        del x_test[elem]
        del y_test[elem]

    x_train, y_train = shuffle(x_train, y_train)
    x_val, y_val = shuffle(x_val, y_val)
    x_test, y_test = shuffle(x_test, y_test)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return x_train, y_train, x_val, y_val, x_test, y_test




#Proper scaling same was as streaming inference
#random sampling of windows in sharkfin region
#Normal = 0, sharkfin = 1, line zero = 2, noise = 3
def gen_labels_multiclass(size, num):
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []


    noise = dict_noise()
    dict_art = dict_artefacts()
    line = dict_zero()


    dict_art_train = {'3-109_1': [], '3-111_1': [], '3-75_1': [], '3-75_2': []}
    dict_art_val = {'3-109_1': [], '3-111_1': [], '3-75_1': [], '3-75_2': []}
    dict_art_test = {'3-109_1': [], '3-111_1': [], '3-75_1': [], '3-75_2': []}

    dict_noise_train = {'3-109_1': [], '3-110_1': [], '3-111_1': [], '3-75_1': [], '3-75_2': [], '3-73_2':[]}
    dict_noise_val = {'3-109_1': [], '3-110_1': [], '3-111_1': [], '3-75_1': [], '3-75_2': [], '3-73_2':[]}
    dict_noise_test = {'3-109_1': [], '3-110_1': [], '3-111_1': [], '3-75_1': [], '3-75_2': [], '3-73_2':[]}

    dict_line_train = {'3-109_1': [], '3-110_1': [], '3-111_1': [], '3-75_1': [], '3-75_2': [], '3-73_2':[]}
    dict_line_val = {'3-109_1': [], '3-110_1': [], '3-111_1': [], '3-75_1': [], '3-75_2': [], '3-73_2':[]}
    dict_line_test = {'3-109_1': [], '3-110_1': [], '3-111_1': [], '3-75_1': [], '3-75_2': [], '3-73_2':[]}


    #len(dict_art[key])
    for key in dict_art.keys():
        dict_art[key] = random.sample(dict_art[key], len(dict_art[key]))
        dict_art_train[key] = dict_art[key][0:int(0.6 * len(dict_art[key]))]
        dict_art_val[key] = dict_art[key][int(0.6 * len(dict_art[key])):int(0.8 * len(dict_art[key]))]
        dict_art_test[key] = dict_art[key][int(0.8 * len(dict_art[key])):]


    for key in line.keys():
        if len(line[key])!=0:
            line[key] = random.sample(line[key], len(line[key]))
            dict_line_train[key] = line[key][0:int(0.6 * len(line[key]))]
            dict_line_val[key] = line[key][int(0.6 * len(line[key])):int(0.8 * len(line[key]))]
            dict_line_test[key] = line[key][int(0.8 * len(line[key])):]

    for key in noise.keys():
        if len(noise[key])!=0:
            noise[key] = random.sample(noise[key], 10)
            dict_noise_train[key] = noise[key][0:int(0.6 * len(noise[key]))]
            dict_noise_val[key] = noise[key][int(0.6 * len(noise[key])):int(0.8 * len(noise[key]))]
            dict_noise_test[key] = noise[key][int(0.8 * len(noise[key])):]

    timebefore = 10  # minutes

    for key in dict_noise_train.keys():
        for value in dict_noise_train[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)

            # get window of the start of artefact region

            windows = rand_windows3(data, num, (timebefore * 60 * 125), size)
            # scale them
            for window in windows:
                if (window[1] - window[0]) < size:
                    continue

                # lis
                lis = data[window[0]:window[1]]

                # 10 min before list
                before = data[window[0] - (timebefore * 60 * 125):window[0]]

                # calculate scaler based on 10 min before
                scaler = StandardScaler()
                reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
                scaler.fit(reshape)

                # scale lis using scaler
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)

                # save
                # append to our dataset
                x_train.append(scaled[0])
                y_train.append(3)

    for key in dict_noise_val.keys():
        for value in dict_noise_val[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)

            # get window of the start of artefact region

            windows = rand_windows3(data, num, (timebefore * 60 * 125), size)
            # scale them
            for window in windows:
                if (window[1] - window[0]) < size:
                    continue

                # lis
                lis = data[window[0]:window[1]]

                # 10 min before list
                before = data[window[0] - (timebefore * 60 * 125):window[0]]

                # calculate scaler based on 10 min before
                scaler = StandardScaler()
                reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
                scaler.fit(reshape)

                # scale lis using scaler
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)

                # save
                # append to our dataset
                x_val.append(scaled[0])
                y_val.append(3)

    for key in dict_noise_test.keys():
        for value in dict_noise_test[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)

            # get window of the start of artefact region

            windows = rand_windows3(data, num, (timebefore * 60 * 125), size)
            # scale them
            for window in windows:
                if (window[1] - window[0]) < size:
                    continue

                # lis
                lis = data[window[0]:window[1]]

                # 10 min before list
                before = data[window[0]-(timebefore*60*125):window[0]]

                # calculate scaler based on 10 min before
                scaler = StandardScaler()
                reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
                scaler.fit(reshape)

                # scale lis using scaler
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)

                # save
                # append to our dataset
                x_test.append(scaled[0])
                y_test.append(3)

####
    for key in dict_line_train.keys():
        for value in dict_line_train[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)

            # get window of the start of artefact region

            windows = rand_windows3(data, num, (timebefore * 60 * 125), size)
            # scale them
            for window in windows:
                if (window[1] - window[0]) < size:
                    continue

                # lis
                lis = data[window[0]:window[1]]

                # 10 min before list
                before = data[window[0]-(timebefore*60*125):window[0]]

                # calculate scaler based on 10 min before
                scaler = StandardScaler()
                reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
                scaler.fit(reshape)

                # scale lis using scaler
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)

                # save
                # append to our dataset
                x_train.append(scaled[0])
                y_train.append(2)

    for key in dict_line_val.keys():
        for value in dict_line_val[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)

            # get window of the start of artefact region

            windows = rand_windows3(data, num, (timebefore * 60 * 125), size)
            # scale them
            for window in windows:
                if (window[1] - window[0]) < size:
                    continue

                # lis
                lis = data[window[0]:window[1]]

                # 10 min before list
                before = data[window[0]-(timebefore*60*125):window[0]]

                # calculate scaler based on 10 min before
                scaler = StandardScaler()
                reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
                scaler.fit(reshape)

                # scale lis using scaler
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)

                # save
                # append to our dataset
                x_val.append(scaled[0])
                y_val.append(2)

    for key in dict_line_test.keys():
        for value in dict_line_test[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)

            # get window of the start of artefact region

            windows = rand_windows3(data, num, (timebefore * 60 * 125), size)
            # scale them
            for window in windows:
                if (window[1] - window[0]) < size:
                    continue

                # lis
                lis = data[window[0]:window[1]]

                # 10 min before list
                before = data[window[0]-(timebefore*60*125):window[0]]

                # calculate scaler based on 10 min before
                scaler = StandardScaler()
                reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
                scaler.fit(reshape)

                # scale lis using scaler
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)

                # save
                # append to our dataset
                x_test.append(scaled[0])
                y_test.append(2)


    for key in dict_art_train.keys():
        for value in dict_art_train[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)

            # get window of the start of artefact region

            windows = rand_windows3(data, num, (timebefore * 60 * 125), size)
            # scale them
            for window in windows:
                if (window[1]-window[0]) < size:
                    continue

                # lis
                lis = data[window[0]:window[1]]

                # 10 min before list
                before = data[window[0]-(timebefore*60*125):window[0]]

                # calculate scaler based on 10 min before
                scaler = StandardScaler()
                reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
                scaler.fit(reshape)

                #scale lis using scaler
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)

                #save
                # append to our dataset
                x_train.append(scaled[0])
                y_train.append(1)

    for key in dict_art_val.keys():
        for value in dict_art_val[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)

            # get window of the start of artefact region

            windows = rand_windows3(data, num, (timebefore * 60 * 125), size)
            # scale them
            for window in windows:
                if (window[1] - window[0]) < size:
                    continue

                # lis
                lis = data[window[0]:window[1]]

                # 10 min before list
                before = data[window[0] - (timebefore * 60 * 125):window[0]]

                # calculate scaler based on 10 min before
                scaler = StandardScaler()
                reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
                scaler.fit(reshape)

                # scale lis using scaler
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)

                # save
                # append to our dataset
                x_val.append(scaled[0])
                y_val.append(1)

    for key in dict_art_test.keys():
        for value in dict_art_test[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*60)), str(value[1])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)

            # get window of the start of artefact region

            windows = rand_windows3(data, num, (timebefore * 60 * 125), size)
            # scale them
            for window in windows:
                if (window[1] - window[0]) < size:
                    continue

                # lis
                lis = data[window[0]:window[1]]

                # 10 min before list
                before = data[window[0] - (timebefore * 60 * 125):window[0]]

                # calculate scaler based on 10 min before
                scaler = StandardScaler()
                reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
                scaler.fit(reshape)

                # scale lis using scaler
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)

                # save
                # append to our dataset
                x_test.append(scaled[0])
                y_test.append(1)

    #GETTING NORMAL DATA
    for key in dict_art_train.keys():
        for value in dict_art_train[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*2*60)), str(value[0])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)

            # get window of the start of artefact region

            windows = rand_windows3(data, 10, (timebefore * 60 * 125), size)
            # scale them
            for window in windows:
                if (window[1]-window[0]) < size:
                    continue

                # lis
                lis = data[window[0]:window[1]]

                # 10 min before list
                before = data[window[0]-(timebefore*60*125):window[0]]

                # calculate scaler based on 10 min before
                scaler = StandardScaler()
                reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
                scaler.fit(reshape)

                #scale lis using scaler
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)

                #save
                # append to our dataset
                x_train.append(scaled[0])
                y_train.append(0)

    for key in dict_art_val.keys():
        for value in dict_art_val[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*2*60)), str(value[0])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)

            # get window of the start of artefact region

            windows = rand_windows3(data, 10, (timebefore * 60 * 125), size)
            # scale them
            for window in windows:
                if (window[1] - window[0]) < size:
                    continue

                # lis
                lis = data[window[0]:window[1]]

                # 10 min before list
                before = data[window[0] - (timebefore * 60 * 125):window[0]]

                # calculate scaler based on 10 min before
                scaler = StandardScaler()
                reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
                scaler.fit(reshape)

                # scale lis using scaler
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)

                # save
                # append to our dataset
                x_val.append(scaled[0])
                y_val.append(0)

    for key in dict_art_test.keys():
        for value in dict_art_test[key]:
            url = str(genURL(str(key), str(value[0]-(timebefore*2*60)), str(value[0])))
            df = genDF(url)

            df = df.drop(columns="Time")
            df = df.values.tolist()
            df = [val for sublist in df for val in sublist]
            data = np.array(df)

            # get window of the start of artefact region

            windows = rand_windows3(data, 10, (timebefore * 60 * 125), size)
            # scale them
            for window in windows:
                if (window[1] - window[0]) < size:
                    continue

                # lis
                lis = data[window[0]:window[1]]

                # 10 min before list
                before = data[window[0] - (timebefore * 60 * 125):window[0]]

                # calculate scaler based on 10 min before
                scaler = StandardScaler()
                reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
                scaler.fit(reshape)

                # scale lis using scaler
                res = lis.reshape(-1, 1)
                scaled = scaler.transform(res)
                scaled = scaled.reshape(1, -1)

                # save
                # append to our dataset
                x_test.append(scaled[0])
                y_test.append(0)

    errors = []
    for i in range(len(x_train)):
        if len(x_train[i]) != size:
            errors.append(i)
    for elem in sorted(errors, reverse=True):
        del x_train[elem]
        del y_train[elem]
    errors = []
    for i in range(len(x_val)):
        if len(x_val[i]) != size:
            errors.append(i)
    for elem in sorted(errors, reverse=True):
        del x_val[elem]
        del y_val[elem]
    errors = []
    for i in range(len(x_test)):
        if len(x_test[i]) != size:
            errors.append(i)
    for elem in sorted(errors, reverse=True):
        del x_test[elem]
        del y_test[elem]

    x_train, y_train = shuffle(x_train, y_train)
    x_val, y_val = shuffle(x_val, y_val)
    x_test, y_test = shuffle(x_test, y_test)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return x_train, y_train, x_val, y_val, x_test, y_test