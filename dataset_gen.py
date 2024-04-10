from hospital_api import *
from tqdm import tqdm 

def main():
    print("Starting Dataset Generation....")
    sizes_seconds = [10, 20, 30, 60, 100, 120] # in seconds
    shifts_seconds = [5, 15, 30]

    sizes = [item*125 for item in sizes_seconds]
    shifts = [item*125 for item in shifts_seconds]

    #generate only start of artefact data
    print("Starting First Type....")
    for size in tqdm(sizes):
        x_train, y_train, x_val, y_val, x_test, y_test = gen_labels9(size)
    
        with open("datasets/abp/start_window_"+str(size)+".pkl", "wb") as f:
            pkl.dump([x_train, y_train, x_val, y_val, x_test, y_test], f)

    # #generate randomly sampled sharkfin region data
    print("Starting Second Type....")
    for size in tqdm(sizes):
        x_train, y_train, x_val, y_val, x_test, y_test = gen_labels10(size)
    
        with open("datasets/abp/random_window_"+str(size)+".pkl", "wb") as f:
            pkl.dump([x_train, y_train, x_val, y_val, x_test, y_test], f)
    #generate start shifted sharfin data
    print("Starting Third Type....")
    for size in tqdm(sizes):
        for shift in shifts:
            if shift < size:
                x_train, y_train, x_val, y_val, x_test, y_test = gen_labels11(size, shift)
    
                with open("datasets/abp/shift_"+str(shift)+"_window_"+str(size)+".pkl", "wb") as f:
                    pkl.dump([x_train, y_train, x_val, y_val, x_test, y_test], f)

if __name__ == "__main__":
    main()