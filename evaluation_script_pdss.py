from evaluation_script_pdss_cvp import *
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from hospital_api import *
import random
from collections import defaultdict

import matplotlib.dates as mdates
import datetime
import csv




""" def plot_tup(tup):
    data=get_data2(160, "3-110_1", tup[0], tup[1])
    plt.plot(data)


def plot_whole(tup):
    data=get_data2(160, "3-110_1", tup[0]-120, tup[1]+120)
    plt.plot(data)

def plot_time(time1, time2):
    data = get_data2(160, "3-110_1", time1, time2)
    plt.plot(data)

def plot_save(tup, i):
    data=get_data2(160, "3-110_1", tup[0]-120, tup[1]+120)
    fig = plt.figure()
    plt.plot(data)
    plt.savefig("tmp/"+str(i) + ".png")
    plt.close(fig)

def inference(tup, window_size, slide_len):
    loaded_model = tensorflow.keras.models.load_model("cnn_1d_1min_nov2.h5")
    #loaded_model._make_predict_function()
    for start_pos in (range(int(tup[0])-60, int(tup[1])+60, slide_len)):
        end_pos = start_pos + window_size
        segment = get_data2(160, "3-110_1", start_pos, end_pos)
        if len(segment)!=window_size*125:
            print(len(segment))
            print("segment too short")
            continue
        #scale - get 10 minutes before the segment
        buffer = get_data2(160, "3-110_1", start_pos-(10*60), start_pos)
        scaler = StandardScaler()
        reshape = buffer.reshape(-1, 1)  # shape that the scaler needs as an input
        scaler.fit(reshape)

        #scale
        res = segment.reshape(-1, 1)
        scaled = scaler.transform(res)
        scaled = scaled.reshape(1, -1)

        ## run inference on scaled segment
        x=scaled.reshape(scaled.shape[0],scaled.shape[1] , 1)
        prob = loaded_model.predict_proba(x)
        print(prob)

def inference_actual(tup, window_size, slide_len):
    loaded_model = tensorflow.keras.models.load_model("cnn_1d_1min_nov2.h5")
    #loaded_model._make_predict_function()

    try:
        buffer = get_data2(160, "3-110_1", tup[0] - (10 * 60), tup[1])

    except:
        print("error")
        return

    scaler = StandardScaler()
    reshape = buffer.reshape(-1, 1)  # shape that the scaler needs as an input
    scaler.fit(reshape)


    # scale
    if len(buffer) != (10+1)*60 * 125:
        segment=buffer[-7500:]
    else:
        segment=buffer[10 * 60 * 125:]

    if len(segment)!=7500:
        print("too short")
        return None
    else:
        res = segment.reshape(-1, 1)
        scaled = scaler.transform(res)
        scaled = scaled.reshape(1, -1)

        ## run inference on scaled segment
        x = scaled.reshape(scaled.shape[0], scaled.shape[1], 1)
        prob = loaded_model.predict_proba(x)
        print(prob)
    return prob
 """

def merge(lis):
    sorted_by_lower_bound = sorted(lis, key=lambda tup: tup[0])
    merged = []

    for higher in sorted_by_lower_bound:
        if not merged:
            merged.append(higher)
        else:
            lower = merged[-1]
            # test for intersection between lower and higher:
            # we know via sorting that lower[0] <= higher[0]
            if higher[0] <= lower[1]:
                upper_bound = max(lower[1], higher[1])
                merged[-1] = (lower[0], upper_bound)  # replace by merged interval
            else:
                merged.append(higher)
    return merged

#Takes dict of {(start,stop): [predictions between start:stop]} and unravels to return a list of [(start,stop)]
def unravel_dict(dict1):
    
    output = []

    for key in dict1:
        if len(dict1[key])!=0:
            for value in dict1[key]:
                output.append(value)
    return output

# with open("retrospect_test_results_slide30.pkl", "rb") as f:
#     TP, TN, FP, FN, missed = pkl.load(f)

def smooth(predictions, window_size, slide_len, method="mean", sigma=1.0):
    if method not in ["mean", "gaussian"]:
        raise ValueError("method should be 'mean' or 'gaussian'")

    smoothed_predictions = []
    # Assuming predictions are sorted by start time.
    i = 0

    # Prepare Gaussian weights if method is Gaussian
    if method == "gaussian":
        weights = gaussian_weights(window_size, sigma=sigma)
    
    while i < len(predictions):
        # Establish the window's time frame.
        window_start, window_end = predictions[i][0], predictions[i][0] + window_size
        window_elements = []
        
        # Collect elements (including score and possibly its index for weighting) within the window.
        while i < len(predictions) and predictions[i][0] < window_end:
            window_elements.append(predictions[i])
            i += 1
        
        # Apply smoothing method
        if window_elements:
            if method == "mean":
                avg_score = sum(score for _, _, score in window_elements) / len(window_elements)
            elif method == "gaussian":
                # Adjust weights based on the actual number of elements in the window
                adjusted_weights = weights[len(weights)//2 - len(window_elements)//2 : len(weights)//2 + len(window_elements)//2 + 1]
                weighted_scores = [elem[2] * w for elem, w in zip(window_elements, adjusted_weights)]
                avg_score = sum(weighted_scores) / sum(adjusted_weights)
            
            smoothed_predictions.append((window_start, window_end, avg_score))
        
        # Move the window by the slide length.
        next_start = predictions[i][0] if i < len(predictions) else window_end
        # Ensure window moves correctly based on slide_len and avoid overlap or gap.
        if next_start - window_end >= slide_len or method == "gaussian":
            i = i  # Stay at current index if next window does not overlap or for Gaussian.
        else:
            i -= len(window_elements) - 1  # Adjust index for overlap.

    return smoothed_predictions

def smooth_discontinuous_sorted(dict1, window_size, slide_len, method="mean"):
    output = []

    for key, predictions in dict1.items():
        # Ensure predictions are sorted.
        predictions_sorted = sorted(predictions, key=lambda x: x[0])
        smoothed_predictions = smooth(predictions_sorted, window_size, slide_len, method)
        output.extend(smoothed_predictions)
    
    # Ensure the final output is sorted by start time.
    output = sorted(output, key=lambda x: x[0])

    return output

def stitch_predictions(smoothed_data, threshold, tolerance=0):
    positive_regions = []
    negative_regions = []

    current_region = None  # Each region is [start, stop, is_positive]
    temp_end = None  # Initialize temp_end to manage temporary end times within tolerance checks

    for (start, stop, sigmoid) in smoothed_data:
        is_positive = sigmoid > threshold
        if current_region is None:
            current_region = [start, stop, is_positive]
        else:
            within_tolerance = start - current_region[1] <= tolerance
            same_type = is_positive == current_region[2]

            if same_type and within_tolerance:
                # Extend the current region or finalize the temp extension.
                current_region[1] = max(stop, temp_end) if temp_end else stop
                temp_end = None  # Reset temp_end after using it.
            elif not same_type and within_tolerance and temp_end is None:
                # Start a temp extension but don't commit yet.
                temp_end = stop
            else:
                # Finalize the current region before starting a new one or after a gap > tolerance.
                if current_region[2]:
                    positive_regions.append((current_region[0], current_region[1]))
                else:
                    negative_regions.append((current_region[0], current_region[1]))
                current_region = [start, stop, is_positive]
                temp_end = None  # Reset temp_end for the new context.

    # Handle the final region after exiting the loop.
    if current_region:
        if current_region[2]:  # Check if the last region is positive.
            positive_regions.append((current_region[0], current_region[1]))
        else:
            negative_regions.append((current_region[0], current_region[1]))

    return positive_regions, negative_regions

def stitch_intervals(intervals, tolerance=0):
    """Stitch intervals with a given tolerance."""
    if not intervals:
        return []
    
    # Sort intervals by start time.
    intervals.sort(key=lambda x: x[0])
    
    stitched_intervals = [intervals[0]]
    for start, stop in intervals[1:]:
        last_stop = stitched_intervals[-1][1]
        if start - last_stop <= tolerance:
            # Extend the last interval if within tolerance.
            stitched_intervals[-1] = (stitched_intervals[-1][0], max(last_stop, stop))
        else:
            # Otherwise, add a new interval.
            stitched_intervals.append((start, stop))
    
    return stitched_intervals


def check_overlap(interval1, interval2, tolerance=0):
    """Check if two intervals overlap, considering an additional tolerance."""
    start1, end1 = interval1
    start2, end2 = interval2
    # Expand the first interval by tolerance
    start1 -= tolerance
    end1 += tolerance
    return max(start1, start2) < min(end1, end2)

def infer_negative_regions(positive_regions, start_time, end_time):
    """Infer negative regions from positive regions given the total observation period."""
    negative_regions = []
    last_end = start_time
    for start, end in sorted(positive_regions):
        if start > last_end:
            negative_regions.append((last_end, start))
        last_end = max(last_end, end)
    if last_end < end_time:
        negative_regions.append((last_end, end_time))
    return negative_regions

def fwhm2sigma(fwhm):
    return fwhm / np.sqrt(8 * np.log(2))


def gaussian_weights(window_size, sigma=1.0):
    """
    Generate Gaussian weights for a given window size.
    """
    radius = window_size // 2
    offset = np.arange(-radius, radius + 1)
    weights = np.exp(-0.5 * (offset / sigma) ** 2)
    weights /= weights.sum()
    return weights



#takes all the results and downsamples predictions to a specified slide
#ie taking a 10s slide and downsampling to a 30s slide involves keeping only every 3rd value
def downsample(lis, current_slide, new_slide):
    output = []
    factor = int(new_slide/current_slide)
    for i in range(0, len(lis), factor):
        output.append(lis[i])

    return output

#takes all the results and downsamples predictions to a specified slide
#ie taking a 10s slide and downsampling to a 30s slide involves keeping only every 3rd value
def downsample_continuous(results, current_slide, new_slide):
    dict = dict_merge(results)

    for key in dict:
        dict[key] = downsample(dict[key], current_slide, new_slide)

    output = []

    for key in dict:
        if len(dict[key])!=0:
            for value in dict[key]:
                output.append(value)
    return output

#takes all the results and downsamples predictions to a specified slide
#ie taking a 10s slide and downsampling to a 30s slide involves keeping only every 3rd value
#Takes as input a sorted list of results with keys being (start,stop) tuple and values being predictions at each slide between start:stop
def downsample_continuous_sorted(dict1, current_slide, new_slide):
    
    if current_slide == new_slide:
        return dict1
    else:
        dict2={}
        for key in dict1:
            dict2[key] = downsample(dict1[key], current_slide, new_slide)

        return dict2

#results is an array where each element: start,stop,prob
def smooth_discontinuous(results, window_size, slide_len, method = "mean"):
    dict = dict_merge(results)

    for key in dict:
        dict[key] = smooth(dict[key], window_size, slide_len, method  = method)

    output = []

    for key in dict:
        if len(dict[key])!=0:
            for value in dict[key]:
                output.append(value)
    return output

def dict_merge(results):
    merged_results = merge(results)

    dict1={}
    for elem in merged_results:
        key = tuple((elem[0],elem[1]))
        dict1[key] = []

    for elem in (sorted(results)):
        for keys in sorted(dict1):
            key = tuple((elem[0], elem[1]))
            if elem[0]>keys[1]:
                continue
            if match(key, keys):
                dict1[keys].append(elem)
    return dict1

#Helper function for continuous regions
def merge_tuple(times):
    saved = list(times[0])
    for st, en in sorted([sorted(t) for t in times]):
        if st <= saved[1]:
            saved[1] = max(saved[1], en)
        else:
            yield tuple(saved)
            saved[0] = st
            saved[1] = en
    yield tuple(saved)

#Helper function for continuous regions
#Takes a list of tuples (start, stop) and returns a list of tuple of continuous, non overlapping intervals
#Allows for a tolerance (ie: if tup_a and tup_b only have time t between them and t<tolerance, then they are merged)
def merge_tuple_tolerance(times, tolerance=0):
    saved = list(times[0])
    for st, en in sorted([sorted(t) for t in times]):
        if  st - saved[1] <= tolerance:
            saved[1] = max(saved[1], en)
        else:
            yield tuple(saved)
            saved[0] = st
            saved[1] = en
    yield tuple(saved)

def merge_intervals(predictions, tolerance=datetime.timedelta(hours=16)):
    """Merge intervals that overlap or are within a specified tolerance."""
    if not predictions:
        return []

    # Sort by start time
    sorted_predictions = sorted(predictions, key=lambda x: x[0])
    merged = [sorted_predictions[0]]

    for current_start, current_end in sorted_predictions[1:]:
        prev_end = merged[-1][1]

        # If the current interval starts before the previous ends or within the tolerance, merge them
        if current_start <= prev_end or (current_start - prev_end) <= tolerance:
            merged[-1] = (merged[-1][0], max(prev_end, current_end))
        else:
            merged.append((current_start, current_end))
    return merged

def plot_predictions(ax, predictions, color, linewidth, solid_capstyle='butt'):
    """Plot predictions on the given axis."""
    for start, stop in predictions:
        ax.plot([start, stop], [1, 1], color=color, linewidth=linewidth, solid_capstyle=solid_capstyle)


# Define lighter pastel colors
pastel_green = '#77dd77'
pastel_red = '#ff6961'

def create_plots(stitched_predictions):
    # Convert and sort all predictions, converting epochs to datetime objects
    all_predictions = [(epoch_to_datetime(start), epoch_to_datetime(stop)) for region in stitched_predictions for start, stop in region]
    
    # Merge overlapping or closely spaced intervals
    merged_predictions = merge_intervals(all_predictions)
    
    # Define plotting specifications
    pos_line_width = 15  # Increased line width for positive predictions
    neg_line_width = 10  # Standard line width for negative predictions
    
    fig, axs = plt.subplots(len(merged_predictions), 1, figsize=(20, 3 * len(merged_predictions)), squeeze=False)
    
    for i, (region_start, region_end) in tqdm(enumerate(merged_predictions), total=len(merged_predictions)):
        ax = axs[i, 0]  # Current axis
        
        # Plot positive and negative predictions within the region
        plot_predictions(ax, stitched_predictions[0], pastel_green, pos_line_width)
        plot_predictions(ax, stitched_predictions[1], pastel_red, neg_line_width)
        
        # Set the limits for the x-axis and format dates
        ax.set_xlim(region_start, region_end)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        
        ax.set_yticks([])  # Remove y-axis
    
    plt.tight_layout()
    plt.show()

#Takes a list of positive predictions with timeoverlaps between elements [(start,stop,prob)]
#returns a list [(start,stop)] where (start, stop) are discrete regions with 0 overlap
#allows for tolerance
def continuous_regions2(predictions, tolerance):
    preds = [(a,b) for (a,b) in predictions]
    first_pass = (list(merge_tuple(preds)))
    return list(merge_tuple_tolerance(first_pass, tolerance))

#Takes a list of positive predictions with timeoverlaps between elements [(start,stop,prob)]
#returns a list [(start,stop)] where (start, stop) are discrete regions with 0 overlap
def continuous_regions(predictions):
    preds = [(a,b) for (a,b,c) in predictions]
    return (list(merge_tuple(preds)))




def check_dict(dict1, window_size, slide_len):
    #TEST TO SEE IF THERE ARE ANY OVERLAPPING KEYS
    overlap_keys = []
    for key1 in dict1.keys():
        for key2 in dict1.keys():
            if match(key1, key2) and key1 != key2:
                overlap_keys.append((key1,key2))
                
    #TEST TO SEE IF WINDOW SIZE OF KEYS IS CORRECT:
    incorrect_window = []
    for key in dict1.keys():
        for (a,b,_) in dict1[key]:
            if b-a != window_size:
                incorrect_window.append((a,b))
                
    #TEST TO SEE IF SLIDE LENGTHS ARE CORRECT:
    #ALSO IMPLICITLY TESTS IF ITEMS ARE IN THE CORRECT ORDER
    incorrect_slide = []
    for key in dict1.keys():
        for i in range(len(dict1[key])-1):
            if dict1[key][i+1][0] - dict1[key][i][0] != slide_len:
                incorrect_slide.append(key)

    #TEST TO SEE IF EACH KEY HAS THE RIGHT NUMBER OF WINDOWS
    incorrect_num_windows = []
    for key in dict1.keys():
        min_value =  dict1[key][0][0] #START time of first item
        max_value = dict1[key][-1][0] #STOP time of last item
        expected_windows = int(((max_value-min_value)/slide_len) + 1)
        if len(dict1[key]) != expected_windows:
            incorrect_num_windows.append(key)
    
    print(overlap_keys, incorrect_window, incorrect_slide, incorrect_num_windows)

def evaluate(pos_predictions, alpha, delta_func, method = "original"):
    ground_truth = gen_ground_truth_clean()

    prec = precision_t(ground_truth, pos_predictions, delta_func, method = method)
    rec = recall_t(ground_truth, pos_predictions, alpha, delta_func)

    f1 = (2*prec*rec)/(prec+rec)

    return prec, rec, f1


def gen_ref_list_int():
    summarydata = pd.read_csv('summarydata_amend.csv')
    summarydata = summarydata.drop(162)
    lis=[]
    for index,row in summarydata.iterrows():
        if str(row["deviceID"])=="3-110_1" or str(row["deviceID"])=="nan":
            start=row["artifact_start"]
            end=row["artifact_finish"]
            tuple=(int(start),int(end))
            lis.append(tuple)
    return lis


def gen_ground_truth_clean_abp():
    with open("ground_truth.pkl", "rb") as f:
        lis = pkl.load(f)
    int_lis=[]
    for elem in lis:
        tuple = (int(elem[0]), int(elem[1]))
        int_lis.append(tuple)
    return int_lis


def gen_ground_truth_clean_cvp():
    artifact_tuples = []

    filenames = ["86_1_cvp.csv", "86_2_cvp.csv"]

    for filename in filenames:
        df=pd.read_csv(filename)

        for item in df["label"].values:
            if isinstance(item, str):
                lis = json.loads(item)
                for elem in lis:
                    if elem["start"] != "NaN" and elem["end"] != "NaN":
                        tup = (int(elem["start"])//1000, int(elem["end"])//1000)
                        artifact_tuples.append(tup)
    return sorted(artifact_tuples)


#takes a csv file of predictions and converts to a list with each item as a prediction (start, stop, sigmoid score)
def load_csv(name):
    data = pd.read_csv(name)
    lis=[]
    for index,row in data.iterrows():
        start=row[0]
        end=row[1]
        sigmoid = row[2]
        tuple=(int(start),int(end), sigmoid)
        lis.append(tuple)
    return lis

# Restrict list from 1467745107 to 1522699657
#Converting S to MS
def restrict_time_range(lis,model):

    if model == "abp":
        new_lis = [(start*1000, stop*1000, sigmoid) for (start, stop, sigmoid) in lis if (start*1000>=1467745107000 and stop*1000<=1522699657000)]
    else:
         new_lis = [(start*1000, stop*1000, sigmoid) for (start, stop, sigmoid) in lis if (start*1000>=1641439671000 and stop*1000<=1656498587000)]
    return new_lis

import pickle as pkl

def sort_results(results, filename):
    dict1 = dict_merge(results)
    with open(filename+"sorted.pkl", "wb") as f:
        pkl.dump(dict1,f)

def sort_all_results():
    datasets=["cnn_1d_30s_start_shift15_3750.csv"]

    for dataset in datasets:
        lis = load_csv("datasets/"+dataset)
        predictions = restrict_time_range(lis)
        sort_results(predictions, dataset)


def retrospective_eval(filepath, filename, alpha, delta_func, precision_method, smooth_method):
    
    print("~~~~EVALUATING " + filename + " ~~~~")
    
    window_size = int(int(filename.split('_')[-2])/125)
    
    with open(filepath+filename, "rb") as f:
        dict1 = pkl.load(f)

    slides = [i for i in range(5,window_size+5,5) if window_size%i == 0]
    precisions =[]
    recalls=[]
    f1s=[]
    thresh = []
    slid = []



    for key in dict1.keys():
        dict1[key] = [(int(a/1000),int(b/1000),c) for (a,b,c) in dict1[key]]

    for slide in (slides):
        if slide<=window_size:


            dict2 = downsample_continuous_sorted(dict1, 5, slide)

            predictions = smooth_discontinuous_sorted(dict2, window_size, slide, smooth_method)

            #thresholds = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.91,0.92,0.93,0.94])
            thresholds = np.array([i/100 for i in range(50,99,1)])


            for threshold in tqdm.tqdm(thresholds):
                pos_predictions = [(int(start), int(stop), sigmoid) for (start, stop, sigmoid) in predictions if sigmoid>threshold]
                if len(pos_predictions)!=0:
                    pos_continuous = continuous_regions(pos_predictions)
                    try:
                        prec, rec, f1 = evaluate(pos_continuous, alpha, delta_func, precision_method)

                        precisions.append(prec)
                        recalls.append(rec)
                        f1s.append(f1)
                        thresh.append(threshold)
                        slid.append(slide)
                    except:
                        continue

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1s = np.array(f1s)
    thresh = np.array(thresh)
    slid = np.array(slid)

    d = {"Threshold": thresh, "precision": precisions, "recall": recalls, "F1_score": f1s,  "Slide":slid}
    df = pd.DataFrame(d)
    
    g = sns.lineplot(data=df, x="Threshold", y="F1_score", hue = "Slide", legend='full', lw=5)
    
    plt.title(filename)
    plt.legend(title='Slide length (seconds)')
    plt.show(g)
    
    return df

def retrospective_all(filepath, alpha, delta_func, precision_method, smooth_method):
    dfs = {}
    
    for filename in os.listdir(filepath):
        f = os.path.join(filepath, filename)
        # checking if it is a file
        if os.path.isfile(f):
            df = retrospective_eval(filepath, filename, alpha, delta_func, precision_method, smooth_method)
            dfs[filename] = df

    return dfs


def evaluate_and_rate_with_intervals(ground_truth_pos, predictions_pos, predictions_neg, start_time, end_time, tolerance=0):
    intervals_TP = []
    intervals_FP = []
    intervals_FN = []
    intervals_TN = []
    missed_durations = []  # Stores the durations of ground truths missed

    matched_gt_pos = [False] * len(ground_truth_pos)

    for pred_interval in predictions_pos:
        found_overlap = False
        for i, gt_interval in enumerate(ground_truth_pos):
            if check_overlap(pred_interval, gt_interval, tolerance):
                intervals_TP.append(pred_interval)
                matched_gt_pos[i] = True
                found_overlap = True
                break
        if not found_overlap:
            intervals_FP.append(pred_interval)

    intervals_initial_FN = [gt_interval for i, gt_interval in enumerate(ground_truth_pos) if not matched_gt_pos[i]]

    for pred_neg_interval in predictions_neg:
        overlaps_with_positive = any(check_overlap(pred_neg_interval, gt_pos, tolerance) for gt_pos in ground_truth_pos)
        if overlaps_with_positive:
            for gt_pos in intervals_initial_FN:
                if check_overlap(pred_neg_interval, gt_pos, tolerance) and gt_pos in intervals_initial_FN:
                    intervals_FN.append(gt_pos)
                    intervals_initial_FN.remove(gt_pos)
        else:
            intervals_TN.append(pred_neg_interval)

    # For each ground truth positive interval not matched (missed), calculate and store its duration
    for gt_missed in intervals_initial_FN:
        start, end = gt_missed
        duration = end - start
        missed_durations.append(duration)  # Store the duration of missed ground truths

    TP, FP, FN, TN = len(intervals_TP), len(intervals_FP), len(intervals_FN), len(intervals_TN)
    TPR = TP / (TP + FN) if TP + FN > 0 else 0
    FPR = FP / (FP + TN) if FP + TN > 0 else 0
    FNR = 1 - TPR
    Precision = TP / (TP + FP) if TP + FP > 0 else 0
    TNR = TN / (TN + FP) if TN + FP > 0 else 0
    Accuracy = (TP + TN) / (TP + FP + FN + TN)

    results = {
        'TPR': TPR, 'FPR': FPR, 'Precision': Precision, 'TNR': TNR, 'FNR': FNR, 'Accuracy': Accuracy,
        'Intervals_TP': intervals_TP, 'Intervals_FP': intervals_FP, 'Intervals_FN': intervals_FN, 'Intervals_TN': intervals_TN,
        'Missed_Durations': len(missed_durations)  # Add missed durations to the results
    }

    return results

def evaluate_and_rate_with_intervals_and_splits(ground_truth_pos, predictions_pos, predictions_neg, tolerance=0):

    # Sort ground truth by start times
    ground_truth_pos_sorted = sorted(ground_truth_pos, key=lambda x: x[0])
    
    # Split the ground truths into 5 equal temporal splits
    splits_indices = np.array_split(np.arange(len(ground_truth_pos_sorted)), 5)
    temporal_splits = [ground_truth_pos_sorted[indices[0]:indices[-1]+1] for indices in splits_indices]
    
    metrics_summary = {
        'TPR': [], 'FPR': [], 'Precision': [], 'TNR': [], 'FNR': [], 'Accuracy': [], "Missed_Durations":[]
    }
    
    for split in temporal_splits:
        # Determine the temporal range of the current split
        start_time, _ = split[0]
        _, end_time = split[-1]
        
        # Filter predictions for the current split
        predictions_pos_filtered = [p for p in predictions_pos if start_time <= p[0] <= end_time or start_time <= p[1] <= end_time]
        predictions_neg_filtered = [p for p in predictions_neg if start_time <= p[0] <= end_time or start_time <= p[1] <= end_time]
        
        # Call the original evaluation function for the current split with the filtered predictions
        results = evaluate_and_rate_with_intervals(split, predictions_pos_filtered, predictions_neg_filtered, start_time, end_time, tolerance)
        
        # Aggregate the results
        for key in metrics_summary:
            metrics_summary[key].append(results[key])
    
    # Calculate mean and standard deviation for each metric across splits
    final_results = {metric: {'mean': np.mean(values), 'std': np.std(values)} for metric, values in metrics_summary.items()}
    
    return final_results

# Note: You need to define `evaluate_and_rate_with_intervals` or integrate its logic directly into the loop for each split.


def generate_latex_table(results_df, metrics, mean_cols, std_cols, caption="Table", label="Label"):
    # Function to format the metric and std values with LaTeX syntax for plus-minus
    def format_value_with_pm(mean, std):
        return f"{mean:.2f} $\\pm$ {std:.2f}".rstrip('0').rstrip('.')
    
    # Ensure the DataFrame has 'Window_Size' for proper labeling
    formatted_rows = []
    for _, row in results_df.iterrows():
        window_size = row['Window_Size']
        formatted_metrics = [f"{window_size}"]
        for mean_col, std_col in zip(mean_cols, std_cols):
            formatted_metrics.append(format_value_with_pm(row[mean_col], row[std_col]))
        formatted_rows.append(" & ".join(formatted_metrics) + " \\\\")

    # Joining all rows
    body = "\n        ".join(formatted_rows)

    # Adjusting header names for LaTeX table
    header_names = [metric.replace('_', ' ').replace('Mean ', '') for metric in metrics]

    # Full LaTeX table format
    table_format = f"""\\begin{{table}}[htbp]
    \\centering
    \\begin{{tabular}}{{{'l' + 'r' * len(metrics)}}}
        \\toprule
        Window Size & {' & '.join(header_names)} \\\\
        \\midrule
        {body}
        \\bottomrule
    \\end{{tabular}}
    \\caption{{{caption}}}
    \\label{{tab:{label}}}
\\end{{table}}
"""

    return table_format

# Convert epoch to datetime
def epoch_to_datetime(epoch):
    return datetime.datetime.utcfromtimestamp(epoch)

#All window sizes, multiple tolerance
def calculate_metrics(model):

    filepath = f"retrospective_results_pdss/{model}/"
    
    # Initialize metrics storage
    metrics = {
        'Window_Size': [],
        'Tolerance': [],
        'Mean_TPR': [], 'Std_TPR': [],
        'Mean_FPR': [], 'Std_FPR': [],
        'Mean_FNR': [], 'Std_FNR': [],
        'Mean_Precision': [], 'Std_Precision': [],
        'Mean_TNR': [], 'Std_TNR': [],
        'Mean_Accuracy': [], 'Std_Accuracy': [],
        'Mean_Missed_Durations': [], 'Std_Missed_Durations': [],
        # Handling missed durations separately if needed
    }

    window_size_to_durations = defaultdict(list)  # For storing missed durations by window size


    for filename in tqdm(os.listdir(filepath)):
        f = os.path.join(filepath, filename)
        # Checking if it is a file
        if os.path.isfile(f) and "random_window" in filename:
            print(filename)
            window_size = int(filename.split('_')[-2]) // 125  # Update parsing as necessary
            original_slide, new_slide = 5,5

            with open(f, "rb") as file:
                dict1 = pkl.load(file)

            for key in dict1.keys():
                dict1[key] = [(int(a / 1000), int(b / 1000), c) for (a, b, c) in dict1[key]]

            # Assuming functions like gen_ground_truth_clean() and smooth_discontinuous_sorted() are already defined
            
            if model == "abp":
                ground_truth = gen_ground_truth_clean_abp()
            else:
                ground_truth = gen_ground_truth_clean_cvp()
            # Adjust ground truth filtering based on your needs
            ground_truth = [(start, stop) for (start, stop) in ground_truth if start > dict1[list(dict1.keys())[0]][0][0] and stop < dict1[list(dict1.keys())[-1]][-1][1]]
            stitched_ground_truth = stitch_intervals(ground_truth)

            threshold = 0.5  # Example threshold

            #for tolerance in range(0, 60, 5):
            tolerance = 30
            predictions = smooth_discontinuous_sorted(dict1, window_size, new_slide, method="gaussian")  # Example prediction processing
            pos_predictions = [(start, stop) for (start, stop, sigmoid) in predictions if sigmoid >= threshold]
            neg_predictions = [(start, stop) for (start, stop, sigmoid) in predictions if sigmoid < threshold]

            # Call the modified evaluation function that handles temporal splits
            results = evaluate_and_rate_with_intervals_and_splits(stitched_ground_truth, pos_predictions, neg_predictions, tolerance)

            # Store metrics, adapting for mean and std storage
            metrics['Window_Size'].append(window_size)
            metrics['Tolerance'].append(tolerance)
            for metric, values in results.items():
                metrics[f'Mean_{metric}'].append(values['mean'])
                metrics[f'Std_{metric}'].append(values['std'])

                # Handling missed durations if necessary

    # Convert metrics dictionary to DataFrame
    df_metrics = pd.DataFrame(metrics)

    # # Save to CSV
    df_metrics.to_csv(f"compiled_metrics_with_splits_{model}.csv", index=False)

def infer_negative_intervals_within_valid(ground_truth_pos, valid_inference_intervals):
    """Infer negative intervals from ground truth within valid inference intervals."""
    negative_intervals = []
    
    for valid_start, valid_stop in valid_inference_intervals:
        # Find ground truth positives that overlap with the current valid interval
        overlapping_positives = [(pos_start, pos_stop) for pos_start, pos_stop in ground_truth_pos if not (pos_stop <= valid_start or pos_start >= valid_stop)]
        
        # Infer negatives within this valid interval by finding gaps between overlapping positives
        if not overlapping_positives:
            # If no positive overlaps, the entire valid interval is negative
            negative_intervals.append((valid_start, valid_stop))
        else:
            # Check for gaps before the first positive, between positives, and after the last positive
            overlapping_positives.sort()
            first_pos_start, _ = overlapping_positives[0]
            _, last_pos_stop = overlapping_positives[-1]

            # Before the first positive
            if first_pos_start > valid_start:
                negative_intervals.append((valid_start, first_pos_start))

            # Between positives
            for (pos_start, pos_stop), (next_pos_start, _) in zip(overlapping_positives, overlapping_positives[1:]):
                if next_pos_start > pos_stop:
                    negative_intervals.append((pos_stop, next_pos_start))

            # After the last positive
            if last_pos_stop < valid_stop:
                negative_intervals.append((last_pos_stop, valid_stop))
    
    return negative_intervals

def get_predictions(model, window_size, threshold = 0.5, tolerance = 60, method = "gaussian"):

    if model == "abp":
        filepath = f"/home/pdss-line-labelling/output/"
    else:
        filepath = f"/home/pdss-line-labelling-cvp/output/"

    for filename in tqdm(os.listdir(filepath)):
        f = os.path.join(filepath, filename)
        # Checking if it is a file
        if os.path.isfile(f) and "random_window_"+str(window_size*125) in filename:
            print(filename)
            window_size = int(filename.split('_')[-1][:-4]) // 125  # Update parsing as necessary
            original_slide, new_slide = 5,5

            # with open(f, "rb") as file:
            #     dict1 = pkl.load(file)

            file = open(f, "r")
            data_list = list(csv.reader(file, delimiter=","))
            file.close()

            data_list = [(int(start[:-4]), int(stop[:-4]), float(sigmoid)) for (start, stop, sigmoid) in data_list]
            data_list = restrict_time_range(data_list, model)
            
            dict1 = dict_merge(data_list)

            for key in dict1.keys():
                dict1[key] = [(int(a / 1000), int(b / 1000), c) for (a, b, c) in dict1[key]]

            # Assuming dict1's keys are (start, stop) tuples in seconds
            valid_inference_intervals = list(dict1.keys())

            if model == "abp":
                ground_truth = gen_ground_truth_clean_abp()
            else:
                ground_truth = gen_ground_truth_clean_cvp()
                
            # Infer negative regions within the entire span of ground truth
            ground_truth_neg = infer_negative_intervals_within_valid(ground_truth, valid_inference_intervals)
            ground_truth_neg = [(int(a / 1000), int(b / 1000)) for (a, b) in ground_truth_neg]

            # Find the time span covered by the data in dict1
            first_predicted_time = dict1[list(dict1.keys())[0]][0][0]
            last_predicted_time = dict1[list(dict1.keys())[-1]][-1][1]

            # Trim positive ground truth regions based on the span covered by dict1
            ground_truth_pos = [(start, stop) for (start, stop) in ground_truth if start >= first_predicted_time and stop <= last_predicted_time]

            # Ensure to cover the span from the first predicted time step to the start of the first positive ground truth region
            if ground_truth_pos:
                first_pos_time = ground_truth_pos[0][0]
                if first_pos_time > first_predicted_time:
                    ground_truth_neg.append((first_predicted_time, first_pos_time))

            # Optionally, sort and merge negative regions if infer_negative_regions doesn't already do it
            # This step might be necessary to ensure the negative regions are properly consolidated
            # ground_truth_neg = merge_intervals(ground_truth_neg, tolerance)

            smoothed_data = smooth_discontinuous_sorted(dict1, window_size, new_slide, method=method)  # Example prediction processing

            stitched_predictions = stitch_predictions(smoothed_data, tolerance=tolerance, threshold=threshold)


    return stitched_predictions, (ground_truth_pos,ground_truth_neg)

def worker_func(filename, directory, output_path, completed_files_no_extension):
    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f) and filename[-8:] not in ["2016.csv", "2017.csv", "2018.csv"] and filename[:-4] not in completed_files_no_extension:
        print("Working on:", f)
        try:
            lis = load_csv(f)
        except:
            return
        if len(lis)!= 0:
            restricted_time = restrict_time_range(lis)
            dict1 = dict_merge(restricted_time)
            with open(output_path+filename[:-4]+"_sorted.pkl", "wb") as f:
                pkl.dump(dict1,f)
            print("Completed:", f)


def main():
    directory = "/home/pdss-line-labelling/output"
    output_path = "/home/PycharmProjects/line_labelling/retrospective_results_pdss/abp/"

    completed_files = [f for f in os.listdir(output_path) if os.path.isfile(os.path.join(output_path, f))]
    completed_files_no_extension = [f[:-11] for f in completed_files]
    # get number of cpus available to job

    n_cores = multiprocessing.cpu_count()
    print(n_cores)

    # created pool running maximum n_cores
    #set_start_method('spawn', force=True)
    pool = Pool(n_cores)
    # Execute the folding task in parallel
    for filename in os.listdir(directory):
        pool.apply_async(worker_func, args=(filename, 
                                            directory, 
                                            output_path, 
                                            completed_files_no_extension))

    # Tell the pool that there are no more tasks to come and join
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()