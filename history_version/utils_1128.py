import os
import pickle
import numpy as np
import pandas as pd
import random
import torch
import math
from torch.autograd import Variable
from helper import *

class DataLoader():

    def __init__(self, f_prefix, batch_size=5, seq_length=20, num_of_validation = 0, forcePreProcess=False, infer=False, generate = False):
        '''
        Initialiser function for the DataLoader class
        params:
        batch_size : Size of the mini-batch
        seq_length : seq_length is not fixed, denponding on the dataset.
        num_of_validation : number of validation dataset will be used
        infer : flag for test mode
        generate : flag for data generation mode
        forcePreProcess : Flag to forcefully preprocess the data again from csv files
        '''
        # base test files and base train files
        base_test_dataset=  []
        base_train_dataset = ['train/test1.csv', 'train/test2.csv']
        
        # TODO Check if this dimensions is needed.
        # dimensions of each file set
        #self.dataset_dimensions = {'biwi':[720, 576], 'crowds':[720, 576], 'stanford':[595, 326], 'mot':[768, 576]}
        
        # List of data directories where raw data resides
        self.base_train_path = 'data/train/'
        self.base_test_path = 'data/test/'
        self.base_validation_path = 'data/validation/'

        # check infer flag, if true choose test directory as base directory
        if infer is False:
            self.base_data_dirs = base_train_dataset
        else:
            self.base_data_dirs = base_test_dataset

        # get all files using python os and base directories
        self.train_dataset = self.get_dataset_path(self.base_train_path, f_prefix)
        self.test_dataset = self.get_dataset_path(self.base_test_path, f_prefix)
        self.validation_dataset = self.get_dataset_path(self.base_validation_path, f_prefix)


        # TODO Check if this generate is needed. 
        # if generate mode, use directly train base files
        #if generate:
        #    self.train_dataset = [os.path.join(f_prefix, dataset[1:]) for dataset in base_train_dataset]

        #request of use of validation dataset
        if num_of_validation > 0:
            self.additional_validation = True
        else:
            self.additional_validation = False

        # check validation dataset availibility and clip the reuqested number if it is bigger than available validation dataset
        if self.additional_validation:
            if len(self.validation_dataset) is 0:
                print("There is no validation dataset.Aborted.")
                self.additional_validation = False
            else:
                num_of_validation = np.clip(num_of_validation, 0, len(self.validation_dataset))
                self.validation_dataset = random.sample(self.validation_dataset, num_of_validation)

        # if not infer mode, use train dataset
        if infer is False:
            self.data_dirs = self.train_dataset
        else:
            # use validation dataset
            if self.additional_validation:
                self.data_dirs = self.validation_dataset
            # use test dataset
            else:
                self.data_dirs = self.test_dataset


        self.infer = infer
        self.generate = generate

        # Number of datasets
        self.numDatasets = len(self.data_dirs)

        # array for keepinng target ships mmsi for each sequence
        self.target_mmsi = []

        # Data directory where the pre-processed pickle file resides
        self.train_data_dir = os.path.join(f_prefix, self.base_train_path)
        self.test_data_dir = os.path.join(f_prefix, self.base_test_path)
        self.val_data_dir = os.path.join(f_prefix, self.base_validation_path)


        # Store the arguments
        self.batch_size = batch_size
        # TODO delete the arg of seq_length, calculate the length of each ship when getting next batch
        self.seq_length = seq_length
        self.orig_seq_lenght = seq_length

        # Validation arguments
        self.val_fraction = 0

        # Define the path in which the process data would be stored
        self.data_file_tr = os.path.join(self.train_data_dir, "cpkl/trajectories_train.cpkl")        
        self.data_file_te = os.path.join(self.base_test_path, "cpkl/trajectories_test.cpkl")
        self.data_file_vl = os.path.join(self.val_data_dir, "cpkl/trajectories_val.cpkl")


        # for creating a dict key: folder names, values: files in this folder
        self.create_folder_file_dict()

        if self.additional_validation:
        # If the file doesn't exist or forcePreProcess is true
            if not(os.path.exists(self.data_file_vl)) or forcePreProcess:
                print("Creating pre-processed validation data from raw data")
                # Preprocess the data from the csv files of the datasets
                # Note that this data is processed in frames
                self.time_preprocess(self.validation_dataset, self.data_file_vl, self.additional_validation)

        if self.infer:
        # if infer mode, and no additional files -> test preprocessing
            if not self.additional_validation:
                if not(os.path.exists(self.data_file_te)) or forcePreProcess:
                    print("Creating pre-processed test data from raw data")
                    # Preprocess the data from the csv files of the datasets
                    # Note that this data is processed in frames
                    print("Working on directory: ", self.data_file_te)
                    self.time_preprocess(self.data_dirs, self.data_file_te)
            # if infer mode, and there are additional validation files -> validation dataset visualization
            else:
                print("Validation visualization file will be created")
        
        # if not infer mode
        else:
            # If the file doesn't exist or forcePreProcess is true -> training pre-process
            if not(os.path.exists(self.data_file_tr)) or forcePreProcess:
                print("Creating pre-processed training data from raw data")
                # Preprocess the data from the csv files of the datasets
                # Note that this data is processed in frames
                self.time_preprocess(self.data_dirs, self.data_file_tr)

        if self.infer:
            # Load the processed data from the pickle file
            if not self.additional_validation: #test mode
                self.load_preprocessed(self.data_file_te)
            else:  # validation mode
                self.load_preprocessed(self.data_file_vl, True)

        else: # training mode
            self.load_preprocessed(self.data_file_tr)

        # Reset all the data pointers of the dataloader object
        self.reset_batch_pointer(valid=False)
        self.reset_batch_pointer(valid=True)

    def time_preprocess(self, data_dirs, data_file, validation_set = False):
        '''
        Function that will pre-process the pixel_pos.csv files of each dataset
        into data with occupancy grid that can be used
        params:
        data_dirs : List of directories where raw data resides
        data_file : The file into which all the pre-processed data needs to be stored
        validation_set: true when a dataset is in validation set
        '''
        # all_timeStamp_data would be a list of list of numpy arrays corresponding to each dataset
        # Each numpy array will correspond to a frame and would be of size (numPeds, 3) each row
        # containing pedID, x, y
        all_timeStamp_data = []
        # Validation frame data
        valid_timeStamp_data = []

        # timeStampList_data would be a list of lists corresponding to each dataset
        # Each list would contain the frameIds of all the frames in the dataset
        timeStampList_data = []
        
        # numShips_data would be a list of lists corresponding to each dataset
        # Ech list would contain the number of pedestrians in each frame in the dataset
        numShips_data = []
        valid_numShips_data = []

        #each list includes ped ids of this frame
        shipsList_data = []
        valid_shipsList_data = []

        # seqLengthList_data would be a list of list of numpy arrays correspnding to each dataset
        # Each numpy array will correspond to a ship sequence length.
        seqLengthList_data = []
        valid_seqLengthList_data = []

        # target ships mmsi for each sequence
        target_mmsi = []
        orig_data = []

        # Index of the current dataset
        dataset_index = 0

        # For each dataset
        for directory in data_dirs:
            # Load the data from the txt file
            print("Now processing: ", directory)
            column_names = ['time_stamp','ship_mmsi','lng','lat']

            # if training mode, read train file to pandas dataframe and process
            if self.infer is False:
                df = pd.read_csv(directory, dtype={'time_stamp':str,'ship_mmsi':str, 'lng':'double', 'lat':'double'}, delimiter = ',',  header=None, names=column_names)
                #drop duplicated ship_mmsi and store the result to target_mmsi
                self.target_mmsi = np.array(df.drop_duplicates(subset={'ship_mmsi'}, keep='first', inplace=False)['ship_mmsi'])

            else:
                # if validation mode, read validation file to pandas dataframe and process
                if self.additional_validation:
                    df = pd.read_csv(directory, dtype={'time_stamp':str,'ship_mmsi':str, 'lng':'double', 'lat':'double'}, delimiter = ' ',  header=None, names=column_names)
                    self.target_mmsi = np.array(df.drop_duplicates(subset={'ship_mmsi'}, keep='first', inplace=False)['ship_mmsi'])

                # if test mode, read test file to pandas dataframe and process
                else:
                    column_names = ['time_stamp','ship_mmsi','lng','lat']
                    df = pd.read_csv(directory, dtype={'time_stamp':str,'ship_mmsi':str, 'lng':'double', 'lat':'double'}, delimiter = ' ',  header=None, names=column_names, converters = {c:lambda x: float('nan') if x == '?' else float(x) for c in ['y','x']})
                    self.target_mmsi = np.array(df[df['y'].isnull()].drop_duplicates(subset={'ship_mmsi'}, keep='first', inplace=False)['ship_mmsi'])

            # convert pandas -> numpy array
            data = np.array(df)
            # keep original copy of file
            orig_data.append(data)

            # reshape the data.
            data = np.swapaxes(data,0,1)

            # get timeStamp List and number of timeStamp
            timeStampList = data[0, :].tolist()
            timeStampList_data.append(timeStampList)
            # Number of timeStamp
            numTimeStamp = len(timeStampList)

            # Initialize the list of numShips for the current dataset
            numShips_data.append([])
            valid_numShips_data.append([])

            # Initialize the list of numpy arrays for the current dataset
            all_timeStamp_data.append([])
            valid_timeStamp_data.append([])

            # list of ships for each frame and seq_length for each ship.
            shipsList_data.append([])
            valid_shipsList_data.append([])

            # Initialize the list of numpy arrays for the current dataset
            seqLengthList_data.append([])
            valid_seqLengthList_data.append([])

            target_mmsi.append(self.target_mmsi)

            seqLengthCount = 0

            for ind, timeStamp in enumerate(timeStampList):
                # Extract all ships in current timeStamp 
                shipsInTimeStamp = data[: , data[0, :] == timeStamp]
                
                # Extract ships list
                shipsList = shipsInTimeStamp[1, :].tolist()

                # Add number of ships in the current timeStamp to the stored data
                shipsWithPos = []

                # For each ship in the current timeStamp
                for ship in shipsList:
                    # Extract their lng and lat positions
                    current_lng = shipsInTimeStamp[3, shipsInTimeStamp[1, :] == ship][0]
                    current_lat = shipsInTimeStamp[2, shipsInTimeStamp[1, :] == ship][0]

                    # Add their ship_mmsi, lng, lat to the row of the numpy array
                    shipsWithPos.append([ship, current_lng, current_lat])

                # At inference time, data generation and if dataset is a validation dataset, no validation data
                if (ind >= numTimeStamp * self.val_fraction) or (self.infer) or (self.generate) or (validation_set):
                    # Add the details of all the peds in the current frame to all_timeStamp_data
                    all_timeStamp_data[dataset_index].append(np.array(shipsWithPos))
                    shipsList_data[dataset_index].append(shipsList)
                    numShips_data[dataset_index].append(len(shipsList))

                    # Count the seq_length of each ship
                    if (ind > 1) and (data[1, ind-1] != data[1, ind]):
                        seqLengthList_data[dataset_index].append(seqLengthCount)
                        seqLengthCount = 0
                    if (ind == numTimeStamp -1):
                        seqLengthList_data[dataset_index].append(seqLengthCount)
                        seqLengthCount = 0
                    seqLengthCount += 1

                else:
                    valid_timeStamp_data[dataset_index].append(np.array(shipsWithPos))
                    valid_shipsList_data[dataset_index].append(shipsList)
                    valid_numShips_data[dataset_index].append(len(shipsList))

                    # Count the seq_length of each ship
                    if (ind > 1) and (data[1, ind-1] != data[1, ind]):
                        valid_seqLengthList_data[dataset_index].append(seqLengthCount)
                        seqLengthCount = 0
                    if (ind == numTimeStamp -1):
                        valid_seqLengthList_data[dataset_index].append(seqLengthCount)
                        seqLengthCount = 0
                    seqLengthCount += 1

            dataset_index += 1
        # Save the arrays in the pickle file
        f = open(data_file, "wb")
        pickle.dump((all_timeStamp_data, timeStampList_data, numShips_data, valid_numShips_data, valid_timeStamp_data, shipsList_data, valid_shipsList_data, seqLengthList_data, valid_seqLengthList_data, target_mmsi, orig_data), f, protocol=2)
        f.close()


    def load_preprocessed(self, data_file, validation_set = False):
        '''
        Function to load the pre-processed data into the DataLoader object
        params:
        data_file : the path to the pickled data file
        validation_set : flag for validation dataset
        '''
        # Load data from the pickled file
        if(validation_set):
            print("Loading validaton datasets: ", data_file)
        else:
            print("Loading train or test dataset: ", data_file)

        f = open(data_file, 'rb')
        self.raw_data = pickle.load(f)
        f.close()

        # Get all the data from the pickle file
        self.data = self.raw_data[0]
        self.timeStampList = self.raw_data[1]
        self.numshipsList = self.raw_data[2]
        self.valid_numshipsList = self.raw_data[3]
        self.valid_data = self.raw_data[4]
        self.shipsList = self.raw_data[5]
        self.valid_shipsList = self.raw_data[6]
        self.seqLengthList = self.raw_data[7]
        self.valid_seqLengthList = self.raw_data[8]
        self.target_mmsi = self.raw_data[9]
        self.orig_data = self.raw_data[10]

        counter = 0
        valid_counter = 0

        # For each dataset
        for dataset in range(len(self.data)):
            # get the frame data for the current dataset
            all_timeStamp_data = self.data[dataset]
            valid_timeStamp_data = self.valid_data[dataset]
            dataset_name = self.data_dirs[dataset].split('/')[-1]
            # calculate number of sequence 
            num_seq_in_dataset = len(self.seqLengthList[dataset])
            num_valid_seq_in_dataset = len(self.valid_seqLengthList[dataset])
            if not validation_set:
                print('Training data from training dataset(name, # time stamp, #sequence)--> ', dataset_name, ':', len(all_timeStamp_data),':', (num_seq_in_dataset))
                print('Validation data from training dataset(name, # time stamp, #sequence)--> ', dataset_name, ':', len(valid_timeStamp_data),':', (num_valid_seq_in_dataset))
            else: 
                print('Validation data from validation dataset(name, # time stamp, #sequence)--> ', dataset_name, ':', len(all_timeStamp_data),':', (num_seq_in_dataset))

            # Increment the counter with the number of sequences in the current dataset
            counter += num_seq_in_dataset
            valid_counter += num_valid_seq_in_dataset

        # Calculate the number of batches
        self.num_batches = int(counter/self.batch_size)
        self.valid_num_batches = int(valid_counter/self.batch_size)


        if not validation_set:
            print('Total number of training batches:', self.num_batches)
            print('Total number of validation batches:', self.valid_num_batches)
        else:
            print('Total number of validation batches:', self.num_batches)

        # self.valid_num_batches = self.valid_num_batches * 2

    def next_batch(self):
        '''
        Function to get the next batch of points
        '''
        # Source data
        x_batch = []
        # Target data
        y_batch = []
        # Dataset data
        d = []

        # num of shiplist per sequence
        numshipsList_batch = []

        # shiplist per sequence
        shipsList_batch = []

        # seq_length of each ship sequence
        seqLengthList_batch = []

        #return target_mmsi
        target_mmsi = []

        # Iteration index
        i = 0
        while i < self.batch_size:
            # Extract the data of the current dataset
            data_dataset = self.data[self.dataset_pointer]
            numshipsList = self.numshipsList[self.dataset_pointer]
            shipsList = self.shipsList[self.dataset_pointer]
            seqLengthList = self.seqLengthList[self.dataset_pointer]
            
            # Get the time stamp pointer for the current dataset
            idx = self.time_pointer
            
            # While there is still seq_length number of frames left in the current dataset
            if self.seqLength_pointer < len(seqLengthList) and idx + seqLengthList[self.seqLength_pointer]-1 < len(data_dataset):
                seq_length = seqLengthList[self.seqLength_pointer]
                # All the data in this sequence
                seq_source_time_data = data_dataset[idx:idx+seq_length]
                seq_numshipsList = numshipsList[idx:idx+seq_length]
                seq_shipsList = shipsList[idx:idx+seq_length]
                seq_target_time_data = data_dataset[idx+1:idx+seq_length+1]

                # Number of unique peds in this sequence of frames
                x_batch.append(seq_source_time_data)
                y_batch.append(seq_target_time_data)
                numshipsList_batch.append(seq_numshipsList)
                shipsList_batch.append(seq_shipsList)
                # get correct target ped id for the sequence
                target_mmsi.append(self.target_mmsi[self.dataset_pointer][self.target_pointer])
                self.time_pointer += seq_length
                self.seqLength_pointer += 1
                self.target_pointer += 1

                d.append(self.dataset_pointer)
                i += 1

            else:
                # Not enough frames left
                # Increment the dataset pointer and set the time_pointer to zero
                self.tick_batch_pointer(valid=False)
        
        return x_batch, y_batch, d, numshipsList_batch, shipsList_batch, target_mmsi


    def next_valid_batch(self):
        '''
        Function to get the next Validation batch of points
        '''
        # Source data
        x_batch = []
        # Target data
        y_batch = []
        # Dataset data
        d = []

        # num of shiplist per sequence
        numshipsList_batch = []

        # shiplist per sequence
        shipsList_batch = []

        # seq_length of each ship sequence
        seqLengthList_batch = []

        #return target_mmsi
        target_mmsi = []

        # Iteration index
        i = 0
        while i < self.batch_size:
            # Extract the frame data of the current dataset
            data_dataset = self.valid_data[self.valid_dataset_pointer]
            numshipsList = self.valid_numshipsList[self.valid_dataset_pointer]
            shipsList = self.valid_shipsList[self.valid_dataset_pointer]
            valid_seqLengthList = self.valid_seqLengthList[self.valid_dataset_pointer]

            # Get the frame pointer for the current dataset
            idx = self.valid_time_pointer
            # While there is still seq_length number of frames left in the current dataset
            #if idx + self.seq_length < len(data_dataset):
            if self.valid_seqLength_pointer < len(valid_seqLengthList) and idx + valid_seqLengthList[self.valid_seqLength_pointer]-1 < len(data_dataset):
                seq_length = valid_seqLengthList[self.valid_seqLength_pointer]
                # All the data in this sequence
                seq_source_time_data = data_dataset[idx:idx+seq_length]
                seq_numshipsList = numshipsList[idx:idx+seq_length]
                seq_shipsList = shipsList[idx:idx+seq_length]
                seq_target_time_data = data_dataset[idx+1:idx+seq_length+1]


                # Number of unique peds in this sequence of frames
                x_batch.append(seq_source_time_data)
                y_batch.append(seq_target_time_data)
                numshipsList_batch.append(seq_numshipsList)
                shipsList_batch.append(seq_shipsList)
                # get correct target ped id for the sequence
                #target_mmsi.append(self.target_mmsi[self.valid_dataset_pointer][math.floor((self.valid_time_pointer)/self.seq_length)])
                target_mmsi.append(self.target_mmsi[self.valid_dataset_pointer][self.valid_target_pointer])
                self.valid_time_pointer += seq_length
                self.valid_time_pointer += seq_length
                self.valid_seqLength_pointer += 1
                self.valid_target_pointer += 1

                d.append(self.valid_dataset_pointer)
                i += 1

            else:
                # Not enough frames left
                # Increment the dataset pointer and set the time_pointer to zero
                self.tick_batch_pointer(valid=True)

        return x_batch, y_batch, d, numshipsList_batch, shipsList_batch, target_mmsi


    def tick_batch_pointer(self, valid=False):
        '''
        Advance the dataset pointer
        '''
        
        if not valid:
            
            # Go to the next dataset
            self.dataset_pointer += 1
            # Set the frame pointer to zero for the current dataset
            self.time_pointer = 0
            self.seqLength_pointer = 0
            self.target_pointer = 0
            # If all datasets are done, then go to the first one again
            if self.dataset_pointer >= len(self.data):
                self.dataset_pointer = 0
            print("*******************")
            print("now processing: %s"% self.get_file_name())
        else:
            # Go to the next dataset
            self.valid_dataset_pointer += 1
            # Set the frame pointer to zero for the current dataset
            self.valid_time_pointer = 0
            # If all datasets are done, then go to the first one again
            if self.valid_dataset_pointer >= len(self.valid_data):
                self.valid_dataset_pointer = 0
            print("*******************")
            print("now processing: %s"% self.get_file_name(pointer_type = 'valid'))

    def reset_batch_pointer(self, valid=False):
        '''
        Reset all pointers
        '''
        if not valid:
            # Go to the first frame of the first dataset
            self.dataset_pointer = 0
            self.time_pointer = 0
            self.seqLength_pointer = 0
            self.target_pointer = 0
        else:
            self.valid_dataset_pointer = 0
            self.valid_time_pointer = 0
            self.valid_seqLength_pointer = 0
            self.valid_target_pointer = 0

    def switch_to_dataset_type(self, train = False, load_data = True):
        # function for switching between train and validation datasets during training session
        print('--------------------------------------------------------------------------')
        if not train: # if train mode, switch to validation mode
            if self.additional_validation:
                print("Dataset type switching: training ----> validation")
                self.orig_seq_lenght, self.seq_length = self.seq_length, self.orig_seq_lenght
                self.data_dirs = self.validation_dataset
                self.numDatasets = len(self.data_dirs)
                if load_data:
                    self.load_preprocessed(self.data_file_vl, True)
                    self.reset_batch_pointer(valid=False)
            else: 
                print("There is no validation dataset.Aborted.")
                return
        else:# if validation mode, switch to train mode
            print("Dataset type switching: validation -----> training")
            self.orig_seq_lenght, self.seq_length = self.seq_length, self.orig_seq_lenght
            self.data_dirs = self.train_dataset
            self.numDatasets = len(self.data_dirs)
            if load_data:
                self.load_preprocessed(self.data_file_tr)
                self.reset_batch_pointer(valid=False)
                self.reset_batch_pointer(valid=True)


    def convert_proper_array(self, x_seq, num_pedlist, pedlist):
        #converter function to appropriate format. Instead of directly use ped ids, we are mapping ped ids to
        #array indices using a lookup table for each sequence -> speed
        #output: seq_lenght (real sequence lenght+1)*max_ped_id+1 (biggest id number in the sequence)*2 (x,y)
        
        #get unique ids from sequence
        unique_ids = pd.unique(np.concatenate(pedlist).ravel().tolist()).astype(str)
        # create a lookup table which maps ped ids -> array indices
        lookup_table = dict(zip(unique_ids, range(0, len(unique_ids))))

        seq_data = np.zeros(shape=(len(x_seq), len(lookup_table), 2))

        # create new structure of array
        for ind, frame in enumerate(x_seq):
            corr_index = [lookup_table[x] for x in frame[:, 0]]
            seq_data[ind, corr_index,:] = frame[:,1:3]

        return_arr = Variable(torch.from_numpy(np.array(seq_data)).float())

        return return_arr, lookup_table

    def add_element_to_dict(self, dict, key, value):
        # helper function to add a element to dictionary
        dict.setdefault(key, [])
        dict[key].append(value)

    def get_dataset_path(self, base_path, f_prefix):
        # get all datasets from given set of directories
        dataset = []
        dir_path = os.path.join(f_prefix, base_path)
        file_names = get_all_file_names(dir_path)
        [dataset.append(os.path.join(dir_path, file_name)) for file_name in file_names]
        return dataset

    def get_file_name(self, offset=0, pointer_type = 'train'):
        #return file name of processing or pointing by dataset pointer
        if pointer_type is 'train':
            return self.data_dirs[self.dataset_pointer+offset].split('/')[-1]
         
        elif pointer_type is 'valid':
            return self.data_dirs[self.valid_dataset_pointer+offset].split('/')[-1]

    def create_folder_file_dict(self):
        # create a helper dictionary folder name:file name
        self.folder_file_dict = {}
        for dir_ in self.base_data_dirs:
            folder_name = dir_.split('/')[-2]
            file_name = dir_.split('/')[-1]
            self.add_element_to_dict(self.folder_file_dict, folder_name, file_name)


    def get_directory_name(self, offset=0):
        #return folder name of file of processing or pointing by dataset pointer
        folder_name = self.data_dirs[self.dataset_pointer+offset].split('/')[-2]
        return folder_name

    def get_directory_name_with_pointer(self, pointer_index):
        # get directory name using pointer index
        # TODO 用windows运行会出现bug，因为linux的/和window的\\
        #folder_name = self.data_dirs[pointer_index].split('/')[-2]
        folder_name = self.data_dirs[pointer_index].split('\\')[-2].split('/')[-1]
        return folder_name

    def get_all_directory_namelist(self):
        #return all directory names in this collection of dataset
        folder_list = [data_dir.split('/')[-2] for data_dir in (self.base_data_dirs)]
        return folder_list

    def get_file_path(self, base, prefix, model_name ='', offset=0):
        #return file path of file of processing or pointing by dataset pointer
        folder_name = self.data_dirs[self.dataset_pointer+offset].split('/')[-2]
        base_folder_name=os.path.join(prefix, base, model_name, folder_name)
        return base_folder_name

    def get_base_file_name(self, key):
        # return file name using folder- file dictionary
        return self.folder_file_dict[key]

    def get_len_of_dataset(self):
        # return the number of dataset in the mode
        return len(self.data)

    def clean_test_data(self, x_seq, target_id, obs_lenght, predicted_lenght):
        #remove (pedid, x , y) array if x or y is nan for each frame in observed part (for test mode)
        for frame_num in range(obs_lenght):
            nan_elements_index = np.where(np.isnan(x_seq[frame_num][:, 2]))

            try:
                x_seq[frame_num] = np.delete(x_seq[frame_num], nan_elements_index[0], axis=0)
            except ValueError:
                print("an error has been occured")
                pass

        for frame_num in range(obs_lenght, obs_lenght+predicted_lenght):
            nan_elements_index = x_seq[frame_num][:, 0] != target_id

            try:
                x_seq[frame_num] = x_seq[frame_num][~nan_elements_index]

            except ValueError:
                pass


    def clean_ped_list(self, x_seq, pedlist_seq, target_id, obs_lenght, predicted_lenght):
        # remove peds from pedlist after test cleaning
        target_id_arr = [target_id]
        for frame_num in range(obs_lenght+predicted_lenght):
            pedlist_seq[frame_num] = x_seq[frame_num][:,0]

    def write_to_file(self, data, base, f_prefix, model_name):
        # write all files as txt format
        self.reset_batch_pointer()
        for file in range(self.numDatasets):
            path = self.get_file_path(f_prefix, base, model_name, file)
            file_name = self.get_file_name(file)
            self.write_dataset(data[file], file_name, path)

    def write_dataset(self, dataset_seq, file_name, path):
        # write a file in txt format
        print("Writing to file  path: %s, file_name: %s"%(path, file_name))
        out = np.concatenate(dataset_seq, axis = 0)
        np.savetxt(os.path.join(path, file_name), out, fmt = "%1d %1.1f %.3f %.3f", newline='\n')

    def write_to_plot_file(self, data, path):
        # write plot file for further visualization in pkl format
        self.reset_batch_pointer()
        for file in range(self.numDatasets):
            file_name = self.get_file_name(file)
            file_name = file_name.split('.')[0] + '.pkl'
            print("Writing to plot file  path: %s, file_name: %s"%(path, file_name))
            with open(os.path.join(path, file_name), 'wb') as f:
                pickle.dump(data[file], f)

    def get_frame_sequence(self, frame_lenght):
        #begin and end of predicted fram numbers in this seq.
        begin_fr = (self.time_pointer - frame_lenght)
        end_fr = (self.time_pointer)
        frame_number = self.orig_data[self.dataset_pointer][begin_fr:end_fr, 0].transpose()
        return frame_number

    def get_id_sequence(self, frame_lenght):
        #begin and end of predicted fram numbers in this seq.
        begin_fr = (self.time_pointer - frame_lenght)
        end_fr = (self.time_pointer)
        id_number = self.orig_data[self.dataset_pointer][begin_fr:end_fr, 1].transpose()
        id_number = [int(i) for i in id_number]
        return id_number

    def get_dataset_dimension(self, file_name):
        # return dataset dimension using dataset file name
        return self.dataset_dimensions[file_name]

