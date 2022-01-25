import os
import pickle
import argparse
import time
import subprocess


import torch
from torch.autograd import Variable

import numpy as np
from utils import DataLoader
from helper import get_mean_error, get_final_error

from criterion import *
from helper import *
from model import getSequenceGridMask


def main():
    
    parser = argparse.ArgumentParser()
    # Model to be loaded
    parser.add_argument('--epoch', type=int, default=40,
                        help='Epoch of model to be loaded')
    
    parser.add_argument('--seq_length', type=int, default=20,
                        help='RNN sequence length')

    parser.add_argument('--use_cuda', action="store_true", default=True,
                        help='Use GPU or not')

    parser.add_argument('--drive', action="store_true", default=False,
                        help='Use Google drive or not')
    # Size of neighborhood to be considered parameter
    parser.add_argument('--neighborhood_size', type=int, default=0.02 * 1000,
                        help='Neighborhood size to be considered for social grid')
    # Size of the social grid parameter
    parser.add_argument('--grid_size', type=int, default=6,
                        help='Grid size of the social grid')
    # number of validation will be used
    parser.add_argument('--num_validation', type=int, default=4,
                        help='Total number of validation dataset will be visualized')
    # gru support
    parser.add_argument('--gru', action="store_true", default=False,
                        help='True : GRU cell, False: LSTM cell')
    # method selection
    parser.add_argument('--method', type=int, default=1,
                        help='Method of lstm will be used (1 = social lstm, 2 = obstacle lstm, 3 = vanilla lstm)')
    
    # Parse the parameters
    sample_args = parser.parse_args()
    
    #for drive run
    prefix = ''
    f_prefix = '.'


    method_name = get_method_name(sample_args.method)
    model_name = "LSTM"
    save_tar_name = method_name+"_lstm_model_"
    if sample_args.gru:
        model_name = "GRU"
        save_tar_name = method_name+"_gru_model_"

    # Save directory
    save_directory = os.path.join(f_prefix, 'model/', method_name, model_name)
    #plot directory for plotting in the future
    plot_directory = os.path.join(f_prefix, 'plot/', method_name, model_name)

    plot_validation_file_directory = 'validation'



    # Define the path for the config file for saved args
    with open(os.path.join(save_directory,'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)

    origin = (0,0)
    reference_point = (0,1)
    net = get_model(sample_args.method, saved_args)
    if sample_args.use_cuda:        
        net = net.cuda()

    # Get the checkpoint path
    checkpoint_path = os.path.join(save_directory, save_tar_name+str(sample_args.epoch)+'.tar')
    if os.path.isfile(checkpoint_path):
        print('Loading checkpoint')
        checkpoint = torch.load(checkpoint_path)
        model_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])
        print('Loaded checkpoint at epoch', model_epoch)

    # Create the DataLoader object
    dataloader = DataLoader(f_prefix, 1, sample_args.seq_length, num_of_validation = sample_args.num_validation, forcePreProcess = True, infer = True)
    create_directories(plot_directory, [plot_validation_file_directory])
    dataloader.reset_batch_pointer()

    print('****************Validation dataset batch processing******************')
    dataloader.reset_batch_pointer(valid=False)
    dataset_pointer_ins = dataloader.dataset_pointer

    loss_epoch = 0
    err_epoch = 0
    f_err_epoch = 0
    num_of_batch = 0
    smallest_err = 100000

    #results of one epoch for all validation datasets
    epoch_result = []
    #results of one validation dataset
    results = []

    # For each batch
    for batch in range(dataloader.num_batches):
        start = time.time()
        # Get batch data
        x, y, d , numPedsList, PedsList ,target_ids = dataloader.next_batch()

        if dataset_pointer_ins is not dataloader.dataset_pointer:
            if dataloader.dataset_pointer is not 0:
                print('Finished prosessed file : ', dataloader.get_file_name(-1),' Avarage error : ', err_epoch/num_of_batch)
                num_of_batch = 0
                epoch_result.append(results)

            dataset_pointer_ins = dataloader.dataset_pointer
            results = []



        # Loss for this batch
        loss_batch = 0
        err_batch = 0
        f_err_batch = 0

        # For each sequence
        for sequence in range(dataloader.batch_size):
            # Get data corresponding to the current sequence
            x_seq ,_ , d_seq, numPedsList_seq, PedsList_seq = x[sequence], y[sequence], d[sequence], numPedsList[sequence], PedsList[sequence]
            target_id = target_ids[sequence]

            #dense vector creation
            x_seq, lookup_seq = dataloader.convert_proper_array(x_seq, numPedsList_seq, PedsList_seq)
            
            #will be used for error calculation
            orig_x_seq = x_seq.clone() 
            
            target_id_values = x_seq[0][lookup_seq[target_id], 0:2]

            #grid mask calculation
            if sample_args.method == 2: #obstacle lstm
                grid_seq = getSequenceGridMask(x_seq, PedsList_seq, saved_args.neighborhood_size, saved_args.grid_size, saved_args.use_cuda, True)
            elif  sample_args.method == 1: #social lstm   
                grid_seq = getSequenceGridMask(x_seq, PedsList_seq, saved_args.neighborhood_size, saved_args.grid_size, saved_args.use_cuda)

            if sample_args.use_cuda:                    
                x_seq = x_seq.cuda()
                orig_x_seq = orig_x_seq.cuda()

            #vectorize datapoints
            x_seq, first_values_dict = vectorize_seq(x_seq, PedsList_seq, lookup_seq)


            if sample_args.method == 3: #vanilla lstm
                ret_x_seq, loss = sample_validation_data_vanilla(x_seq, PedsList_seq, sample_args, net, lookup_seq, numPedsList_seq, dataloader)

            else:
                ret_x_seq, loss = sample_validation_data(x_seq, PedsList_seq, grid_seq, sample_args, net, lookup_seq, numPedsList_seq, dataloader)
            
            if loss == 0:
                continue


            #revert the points back to original space
            ret_x_seq = revert_seq(ret_x_seq, PedsList_seq, lookup_seq, first_values_dict)
            
            err = get_mean_error(ret_x_seq.data, orig_x_seq.data, PedsList_seq, PedsList_seq, sample_args.use_cuda, lookup_seq)
            f_err = get_final_error(ret_x_seq.data, orig_x_seq.data, PedsList_seq, PedsList_seq, lookup_seq)
           

            loss_batch += loss.item()
            err_batch += err
            f_err_batch += f_err
            results.append((orig_x_seq.data.cpu().numpy(), ret_x_seq.data.cpu().numpy(), PedsList_seq, lookup_seq, dataloader.get_frame_sequence(sample_args.seq_length), target_id))
        
            end = time.time()
            loss_batch = loss_batch / dataloader.batch_size
            err_batch = err_batch / dataloader.batch_size
            f_err_batch = f_err_batch / dataloader.batch_size
            print('Current file : ', dataloader.get_file_name(0),' Batch : ', batch+1, ' mean error: ', err_batch ,' final error: ',f_err_batch ,' time: ', end - start)
            num_of_batch += 1
            loss_epoch += loss_batch
            err_epoch += err_batch
            f_err_epoch += f_err_batch

    epoch_result.append(results)


    if dataloader.num_batches != 0:            
        loss_epoch = loss_epoch / dataloader.num_batches
        err_epoch = err_epoch / dataloader.num_batches
        f_err_epoch = f_err_epoch / dataloader.num_batches
        print('valid_loss = {:f}, valid_mean_err = {:f}, valid_final_err = {:f}'.format(loss_epoch, err_epoch, f_err_epoch))

    dataloader.write_to_plot_file(epoch_result, os.path.join(plot_directory, plot_validation_file_directory))


if __name__ == '__main__':
    main()


