from hospital_api import *
import json
import os

from evaluation_script_cvp import *
from tqdm import tqdm 

def artefact_summary_cvp():
    artifact_start = []
    artifact_finish = []
    labels = []
    device = []
    
    filenames = ["84_cvp.csv", "85_cvp.csv", "87_cvp.csv"]
    
    for filename in filenames:
        df=pd.read_csv(filename)

        for item in df["label"].values:
            if isinstance(item, str):
                lis = json.loads(item)
                for elem in lis:
                    if elem["start"] != "NaN" and elem["end"] != "NaN":
                        artifact_start.append(int(elem["start"]))
                        artifact_finish.append(int(elem["end"]))
                        labels.append("artefact")
                        device.append(filename[:2])

    final_df = {"artifact_start": artifact_start, "artifact_finish": artifact_finish, "label":labels, "deviceID":device}

    final_df = pd.DataFrame.from_dict(final_df)

    return final_df

def noise_summary_cvp():
    artifact_start = []
    artifact_finish = []
    labels = []
    device = []
    
    filenames = ["84_cvp.csv", "85_cvp.csv", "87_cvp.csv"]
    
    for filename in filenames:
        df=pd.read_csv(filename)

        for item in df["label2"].values:
            if isinstance(item, str):
                lis = json.loads(item)
                for elem in lis:
                    if elem["start"] != "NaN" and elem["end"] != "NaN":
                        artifact_start.append(int(elem["start"]))
                        artifact_finish.append(int(elem["end"]))
                        labels.append("noise")
                        device.append(filename[:2])

    final_df = {"artifact_start": artifact_start, "artifact_finish": artifact_finish, "label":labels, "deviceID":device}

    final_df = pd.DataFrame.from_dict(final_df)

    return final_df

def remove_overlaps(list_tuples):
    result = [list_tuples[0]]

    bounds = list_tuples[0][0], list_tuples[0][1]

    for tup in list_tuples[1:]:
        if tup[0] in range(bounds[0], bounds[1]) or tup[1] in range(bounds[0], bounds[1]):
            pass
        else:
            result.append(tup)
    return result

def dict_artefacts_cvp():
    dict={'84':[], '85':[], '87':[]}
    pd = artefact_summary_cvp()
    for index, row in pd.iterrows():
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
        
    #ensure no overlaps
    for key in dict.keys():
        dict[key] = remove_overlaps(dict[key])
    return dict

def dict_noise_cvp():
    dict={'84':[], '85':[], '87':[]}
    pd = noise_summary_cvp()
    for index, row in pd.iterrows():
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
        
    #ensure no overlaps
    for key in dict.keys():
        dict[key] = remove_overlaps(dict[key])
    return dict

#Given a tuple of time steps and a device ID, we pull the correct data

#helper function, find the right files to pull


#Load_CSV
def read_csv_and_do_something(csv_filename):
    df = pd.read_csv(csv_filename)
    df["Pressure"]=df["values"]
    df["Time"] = df["times"]//1000000
    df = df.set_index('Time')
    return df


def merge_overlapping_df(df1, df2):
    df3 = (df1
          .combine_first(df2)
          #.reset_index()
      ) 
    return df3

def files_to_df(filepath, device, files_to_grab):
    df1 = read_csv_and_do_something(filepath+device+"/"+files_to_grab[0])
    df2 = read_csv_and_do_something(filepath+device+"/"+files_to_grab[1])
    
    return merge_overlapping_df(df1, df2)

def get_df_csv(filepath, device, start, stop):
    #List of All files in directory
    arr = sorted(os.listdir(filepath+device+"/"))
    arr = [int(x[:-4])//1000000 for x in arr]

    #identify correct files to grab
    files_to_grab = []
    for i in range(0, len(arr)-1):
        if start >= arr[i] and stop <=arr[i+1]:
            #files_to_grab.append('{0:f}'.format(int(arr[i-1]*1000000000))[:-7]+".csv")
            files_to_grab.append(str(int(arr[i]*1000000))+".csv")
            files_to_grab.append(str(int(arr[i+1]*1000000))+".csv")
            
    if len(files_to_grab)!=0:
        return files_to_df(filepath, device, files_to_grab)
    else:
        return None

def get_df_csv_sec(filepath, device, start, stop):
    #List of All files in directory
    arr = sorted(os.listdir(filepath+device+"/"))
    arr = [int(x[:-4])//1000000 for x in arr]

    #identify correct files to grab
    files_to_grab = []
    for i in range(0, len(arr)-1):
        if start*1000 >= arr[i] and stop*1000 <=arr[i+1]:
            #files_to_grab.append('{0:f}'.format(int(arr[i-1]*1000000000))[:-7]+".csv")
            files_to_grab.append(str(int(arr[i]*1000000))+".csv")
            files_to_grab.append(str(int(arr[i+1]*1000000))+".csv")
            
    if len(files_to_grab)!=0:
        return files_to_df(filepath, device, files_to_grab)
    else:
        return None


def plot_window(start, stop):
    filepath = "/home/sujay/cvp_data/raw_data/"
    device = "86"
    
    #print("Duration (s):", stop-start)
    df = get_df_csv_sec(filepath, device, start, stop)
    plt.figure()
    
    plt.plot(df.loc[start*1000:stop*1000]["Pressure"].values)



def dict_normal_cvp():
    noise = dict_noise_cvp()
    dict_art = dict_artefacts_cvp()
    combined = {}
    output = {}
    for key in noise:
        combined[key] = sorted(noise[key]+dict_art[key])
        merged = dict_merge(combined[key])
        keys = list(merged.keys())
        
        if len(keys)%2 != 0:
            keys = keys[:-1]
        
        n = 2
        new = [keys[i * n:(i + 1) * n] for i in range((len(keys) + n - 1) // n )] 
        
        output[key] = [(b,c) for [(a,b),(c,d)] in new]
    return output

def gen_labels10_cvp(size):
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []

    filepath = "/home/sujay/cvp_data/raw_data/"

    noise = dict_noise_cvp()
    dict_art = dict_artefacts_cvp()
    normal = dict_normal_cvp()



    dict_art_train = {'84':[], '85':[], '87':[]}
    dict_art_val = {'84':[], '85':[], '87':[]}
    dict_art_test = {'84':[], '85':[], '87':[]}

    dict_noise_train = {'84':[], '85':[], '87':[]}
    dict_noise_val = {'84':[], '85':[], '87':[]}
    dict_noise_test = {'84':[], '85':[], '87':[]}
    
    dict_normal_train = {'84':[], '85':[], '87':[]}
    dict_normal_val = {'84':[], '85':[], '87':[]}
    dict_normal_test = {'84':[], '85':[], '87':[]}



    #SPLITTING DATA AT THE LEVEL OF THE ARTEFACT, LINE ZERO, NOISE
    #So that the windows from the same artefact are not found in more than one of train, val, or test
    for key in dict_art.keys():
        dict_art[key] = random.sample(dict_art[key], len(dict_art[key]))
        #dict_art[key] = random.sample(dict_art[key], 100)
        dict_art_train[key] = dict_art[key][0:int(0.6 * len(dict_art[key]))]
        dict_art_val[key] = dict_art[key][int(0.6 * len(dict_art[key])):int(0.8 * len(dict_art[key]))]
        dict_art_test[key] = dict_art[key][int(0.8 * len(dict_art[key])):]

    for key in noise.keys():
        if len(noise[key])!=0:
            noise[key] = random.sample(noise[key], len(noise[key]))
            #noise[key] = random.sample(noise[key], 100)
            dict_noise_train[key] = noise[key][0:int(0.6 * len(noise[key]))]
            dict_noise_val[key] = noise[key][int(0.6 * len(noise[key])):int(0.8 * len(noise[key]))]
            dict_noise_test[key] = noise[key][int(0.8 * len(noise[key])):]
    
    for key in normal.keys():
        if len(normal[key])!=0:
            normal[key] = random.sample(normal[key], len(normal[key]))
            #normal[key] = random.sample(normal[key], 100)
            dict_normal_train[key] = normal[key][0:int(0.6 * len(normal[key]))]
            dict_normal_val[key] = normal[key][int(0.6 * len(normal[key])):int(0.8 * len(normal[key]))]
            dict_normal_test[key] = normal[key][int(0.8 * len(normal[key])):]

    timebefore = 10  # minutes

    for key in (dict_noise_train.keys()):
        for value in tqdm(dict_noise_train[key]):
            start,stop = value

            #Get 10 minutes before window start to end of window
            df = get_df_csv(filepath, key, start, stop)
            try:
                data = df.loc[start-(timebefore*60*1000):stop]["Pressure"].values 
            except:
                continue

            # get slices of the region
            max_windows = (len(data[(timebefore * 60 * 125):]))//(size) 
            windows = rand_windows3(data, 1, (timebefore * 60 * 125), size)
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

    for key in (dict_noise_val.keys()):
        for value in tqdm(dict_noise_val[key]):
            start,stop = value

            #Get 10 minutes before window start to end of window
            df = get_df_csv(filepath, key, start, stop)
            try:
                data = df.loc[start-(timebefore*60*1000):stop]["Pressure"].values 
            except:
                continue
            # get slices of the region

            max_windows = (len(data[(timebefore * 60 * 125):]))//(size) 
            windows = rand_windows3(data, 1, (timebefore * 60 * 125), size)
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

    for key in (dict_noise_test.keys()):
        for value in tqdm(dict_noise_test[key]):
            start,stop = value

            #Get 10 minutes before window start to end of window
            df = get_df_csv(filepath, key, start, stop)
            try:
                data = df.loc[start-(timebefore*60*1000):stop]["Pressure"].values 
            except:
                continue
                # get slices of the region

            max_windows = (len(data[(timebefore * 60 * 125):]))//(size) 
            windows = rand_windows3(data, 1, (timebefore * 60 * 125), size)
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
        
        
    for key in (dict_normal_train.keys()):
        for value in tqdm(dict_normal_train[key]):
            start,stop = value

            #Get 10 minutes before window start to end of window
            df = get_df_csv(filepath, key, start, stop)
            try:
                data = df.loc[start-(timebefore*60*1000):stop]["Pressure"].values 
            except:
                continue

            # get slices of the region
            max_windows = (len(data[(timebefore * 60 * 125):]))//(size) 
            windows = rand_windows3(data, 5, (timebefore * 60 * 125), size)
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

    for key in (dict_normal_val.keys()):
        for value in tqdm(dict_normal_val[key]):
            start,stop = value

            #Get 10 minutes before window start to end of window
            df = get_df_csv(filepath, key, start, stop)
            try:
                data = df.loc[start-(timebefore*60*1000):stop]["Pressure"].values 
            except:
                continue
            # get slices of the region

            max_windows = (len(data[(timebefore * 60 * 125):]))//(size) 
            windows = rand_windows3(data, 5, (timebefore * 60 * 125), size)
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

    for key in (dict_normal_test.keys()):
        for value in tqdm(dict_normal_test[key]):
            start,stop = value

            #Get 10 minutes before window start to end of window
            df = get_df_csv(filepath, key, start, stop)
            try:
                data = df.loc[start-(timebefore*60*1000):stop]["Pressure"].values 
            except:
                continue
                # get slices of the region

            max_windows = (len(data[(timebefore * 60 * 125):]))//(size) 
            windows = rand_windows3(data, 5, (timebefore * 60 * 125), size)
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

    for key in (dict_art_train.keys()):
        for value in tqdm(dict_art_train[key]):
            start,stop = value

            #Get 10 minutes before window start to end of window
            df = get_df_csv(filepath, key, start, stop)
            try:
                data = df.loc[start-(timebefore*60*1000):stop]["Pressure"].values 
            except:
                continue
                # get slices of the region

            max_windows = (len(data[(timebefore * 60 * 125):]))//(size) 
            windows = rand_windows3(data, max_windows, (timebefore * 60 * 125), size)
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

    for key in (dict_art_val.keys()):
        for value in tqdm(dict_art_val[key]):
            start,stop = value

            #Get 10 minutes before window start to end of window
            df = get_df_csv(filepath, key, start, stop)
            try:
                data = df.loc[start-(timebefore*60*1000):stop]["Pressure"].values 
            except:
                continue
                # get slices of the region

            max_windows = (len(data[(timebefore * 60 * 125):]))//(size) 
            windows = rand_windows3(data, max_windows, (timebefore * 60 * 125), size)
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
                x_val.append(scaled[0])
                y_val.append(1)

    for key in (dict_art_test.keys()):
        for value in tqdm(dict_art_test[key]):
            start,stop = value

            #Get 10 minutes before window start to end of window
            df = get_df_csv(filepath, key, start, stop)
            try:
                data = df.loc[start-(timebefore*60*1000):stop]["Pressure"].values 
            except:
                continue

            # get slices of the region

            max_windows = (len(data[(timebefore * 60 * 125):]))//(size) 
            windows = rand_windows3(data, max_windows, (timebefore * 60 * 125), size)
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
def gen_labels9_cvp(size):
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []

    filepath = "/home/sujay/cvp_data/raw_data/"

    noise = dict_noise_cvp()
    dict_art = dict_artefacts_cvp()



    dict_art_train = {'84':[], '85':[], '87':[]}
    dict_art_val = {'84':[], '85':[], '87':[]}
    dict_art_test = {'84':[], '85':[], '87':[]}

    dict_noise_train = {'84':[], '85':[], '87':[]}
    dict_noise_val = {'84':[], '85':[], '87':[]}
    dict_noise_test = {'84':[], '85':[], '87':[]}



    #SPLITTING DATA AT THE LEVEL OF THE ARTEFACT, LINE ZERO, NOISE
    #So that the windows from the same artefact are not found in more than one of train, val, or test
    for key in dict_art.keys():
        dict_art[key] = random.sample(dict_art[key], len(dict_art[key]))
        #dict_art[key] = random.sample(dict_art[key], 10)
        dict_art_train[key] = dict_art[key][0:int(0.6 * len(dict_art[key]))]
        dict_art_val[key] = dict_art[key][int(0.6 * len(dict_art[key])):int(0.8 * len(dict_art[key]))]
        dict_art_test[key] = dict_art[key][int(0.8 * len(dict_art[key])):]

    for key in noise.keys():
        if len(noise[key])!=0:
            noise[key] = random.sample(noise[key], len(noise[key]))
            #noise[key] = random.sample(noise[key], 10)
            dict_noise_train[key] = noise[key][0:int(0.6 * len(noise[key]))]
            dict_noise_val[key] = noise[key][int(0.6 * len(noise[key])):int(0.8 * len(noise[key]))]
            dict_noise_test[key] = noise[key][int(0.8 * len(noise[key])):]

    timebefore = 10  # minutes

    for key in dict_noise_train.keys():
        
        for value in dict_noise_train[key]:
            start,stop = value

            #Get 10 minutes before window start to end of window
            df = get_df_csv(filepath, key, start, stop)
            try:
                data = df.loc[start-(timebefore*60*1000):stop]["Pressure"].values 
            except:
                continue
                # get slices of the region

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

    for key in dict_noise_val.keys():
        for value in dict_noise_val[key]:
            start,stop = value

            #Get 10 minutes before window start to end of window
            df = get_df_csv(filepath, key, start, stop)
            try:
                data = df.loc[start-(timebefore*60*1000):stop]["Pressure"].values 
            except:
                continue
                # get slices of the region

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

    for key in dict_noise_test.keys():
        for value in dict_noise_test[key]:
            start,stop = value

            #Get 10 minutes before window start to end of window
            df = get_df_csv(filepath, key, start, stop)
            try:
                data = df.loc[start-(timebefore*60*1000):stop]["Pressure"].values 
            except:
                continue
                # get slices of the region

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


    for key in dict_art_train.keys():
        for value in dict_art_train[key]:
            start,stop = value

            #Get 10 minutes before window start to end of window
            df = get_df_csv(filepath, key, start, stop)
            try:
                data = df.loc[start-(timebefore*60*1000):stop+(size//125*1000)]["Pressure"].values 
            except:
                continue

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
            x_train.append(scaled[0])
            y_train.append(1)

    for key in dict_art_val.keys():
        for value in dict_art_val[key]:
            start,stop = value

            #Get 10 minutes before window start to end of window
            df = get_df_csv(filepath, key, start, stop)
            try:
                data = df.loc[start-(timebefore*60*1000):stop+(size//125*1000)]["Pressure"].values 
            except:
                continue

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
            start,stop = value

            #Get 10 minutes before window start to end of window
            df = get_df_csv(filepath, key, start, stop)
            try:
                data = df.loc[start-(timebefore*60*1000):stop+(size//125*1000)]["Pressure"].values 
            except:
                continue

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

##only start of artefact
##Random perturbation of start
def gen_labels11_cvp(size, shift):
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []

    filepath = "/home/sujay/cvp_data/raw_data/"

    noise = dict_noise_cvp()
    dict_art = dict_artefacts_cvp()



    dict_art_train = {'84':[], '85':[], '87':[]}
    dict_art_val = {'84':[], '85':[], '87':[]}
    dict_art_test = {'84':[], '85':[], '87':[]}

    dict_noise_train = {'84':[], '85':[], '87':[]}
    dict_noise_val = {'84':[], '85':[], '87':[]}
    dict_noise_test = {'84':[], '85':[], '87':[]}



    #SPLITTING DATA AT THE LEVEL OF THE ARTEFACT, LINE ZERO, NOISE
    #So that the windows from the same artefact are not found in more than one of train, val, or test
    for key in dict_art.keys():
        dict_art[key] = random.sample(dict_art[key], len(dict_art[key]))
        #dict_art[key] = random.sample(dict_art[key], 10)
        dict_art_train[key] = dict_art[key][0:int(0.6 * len(dict_art[key]))]
        dict_art_val[key] = dict_art[key][int(0.6 * len(dict_art[key])):int(0.8 * len(dict_art[key]))]
        dict_art_test[key] = dict_art[key][int(0.8 * len(dict_art[key])):]

    for key in noise.keys():
        if len(noise[key])!=0:
            noise[key] = random.sample(noise[key], len(noise[key]))
            #noise[key] = random.sample(noise[key], 10)
            dict_noise_train[key] = noise[key][0:int(0.6 * len(noise[key]))]
            dict_noise_val[key] = noise[key][int(0.6 * len(noise[key])):int(0.8 * len(noise[key]))]
            dict_noise_test[key] = noise[key][int(0.8 * len(noise[key])):]

    timebefore = 10  # minutes

    for key in dict_noise_train.keys():
        
        for value in dict_noise_train[key]:
            start,stop = value

            #Get 10 minutes before window start to end of window
            df = get_df_csv(filepath, key, start, stop)
            try:
                data = df.loc[start-(timebefore*60*1000):stop]["Pressure"].values 
            except:
                continue
                # get slices of the region

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

    for key in dict_noise_val.keys():
        for value in dict_noise_val[key]:
            start,stop = value

            #Get 10 minutes before window start to end of window
            df = get_df_csv(filepath, key, start, stop)
            try:
                data = df.loc[start-(timebefore*60*1000):stop]["Pressure"].values 
            except:
                continue
                # get slices of the region

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

    for key in dict_noise_test.keys():
        for value in dict_noise_test[key]:
            start,stop = value

            #Get 10 minutes before window start to end of window
            df = get_df_csv(filepath, key, start, stop)
            try:
                data = df.loc[start-(timebefore*60*1000):stop]["Pressure"].values 
            except:
                continue
                # get slices of the region

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


    for key in dict_art_train.keys():
        for value in dict_art_train[key]:
            start,stop = value

            #Get 10 minutes before window start to end of window
            df = get_df_csv(filepath, key, start, stop)
            try:
                shifted_start = start + random.randint(-shift//125*1000, 0)
                data = df.loc[shifted_start-(timebefore*60*1000):stop+(size//125*1000)]["Pressure"].values 
            except:
                continue

            if len(data)<timebefore*60*125:
                continue

            before = data[:timebefore*60*125]
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
            x_train.append(scaled[0])
            y_train.append(1)

    for key in dict_art_val.keys():
        for value in dict_art_val[key]:
            start,stop = value

            #Get 10 minutes before window start to end of window
            df = get_df_csv(filepath, key, start, stop)
            try:
                shifted_start = start + random.randint(-shift//125*1000, 0)
                data = df.loc[shifted_start-(timebefore*60*1000):stop+(size//125*1000)]["Pressure"].values 
            except:
                continue

            if len(data)<timebefore*60*125:
                continue

            before = data[:timebefore*60*125]
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
            start,stop = value

            #Get 10 minutes before window start to end of window
            df = get_df_csv(filepath, key, start, stop)
            try:
                shifted_start = start + random.randint(-shift//125*1000, 0)
                data = df.loc[shifted_start-(timebefore*60*1000):stop+(size//125*1000)]["Pressure"].values 
            except:
                continue

            if len(data)<timebefore*60*125:
                continue

            before = data[:timebefore*60*125]
            scaler = StandardScaler()
            reshape = before.reshape(-1, 1)  # shape that the scaler needs as an input
            scaler.fit(reshape)
            # get window of the start of artefact region

            window = data[timebefore * 60 * 125:(timebefore * 60 * 125)+size]
        # scale them
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