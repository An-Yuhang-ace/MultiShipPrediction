import numpy as np
import torch
from torch.autograd import Variable

import os
import shutil
from os import walk
import math
import pandas as pd

from model import SocialModel, OLSTMModel, VLSTMModel
from criterion import *


#one time set dictionary for a exist key
class WriteOnceDict(dict):
    def __setitem__(self, key, value):
        if not key in self:
            super(WriteOnceDict, self).__setitem__(key, value)

#(1 = social lstm, 2 = obstacle lstm, 3 = vanilla lstm)
def get_method_name(index):
    # return method name given index
    return {
        1 : 'SOCIALLSTM',
        2 : 'OBSTACLELSTM',
        3 : 'VANILLALSTM'
    }.get(index, 'SOCIALLSTM')

def get_model(index, arguments, infer = False):
    # return a model given index and arguments
    if index == 1:
        return SocialModel(arguments, infer)
    elif index == 2:
        return OLSTMModel(arguments, infer)
    elif index == 3:
        return VLSTMModel(arguments, infer)
    else:
        return SocialModel(arguments, infer)

def getCoef(outputs):
    '''
    Extracts the mean, standard deviation and correlation
    params:
    outputs : Output of the SRNN model
    '''
    mux, muy, sx, sy, corr = outputs[:, :, 0], outputs[:, :, 1], outputs[:, :, 2], outputs[:, :, 3], outputs[:, :, 4]

    sx = torch.exp(sx)
    sy = torch.exp(sy)
    corr = torch.tanh(corr)
    return mux, muy, sx, sy, corr


def sample_gaussian_2d(mux, muy, sx, sy, corr, nodesPresent, look_up):
    '''
    Parameters
    ==========

    mux, muy, sx, sy, corr : a tensor of shape 1 x numNodes
    Contains x-means, y-means, x-stds, y-stds and correlation

    nodesPresent : a list of nodeIDs present in the frame
    look_up : lookup table for determining which ped is in which array index

    Returns
    =======

    next_x, next_y : a tensor of shape numNodes
    Contains sampled values from the 2D gaussian
    '''
    o_mux, o_muy, o_sx, o_sy, o_corr = mux[0, :], muy[0, :], sx[0, :], sy[0, :], corr[0, :]

    numNodes = mux.size()[1]
    next_x = torch.zeros(numNodes)
    next_y = torch.zeros(numNodes)
    converted_node_present = [look_up[node] for node in nodesPresent]
    for node in range(numNodes):
        if node not in converted_node_present:
            continue
        mean = [o_mux[node].cpu(), o_muy[node].cpu()]
        cov = [[o_sx[node].cpu()*o_sx[node].cpu(), o_corr[node].cpu()*o_sx[node].cpu()*o_sy[node].cpu()], 
                [o_corr[node].cpu()*o_sx[node].cpu()*o_sy[node].cpu(), o_sy[node].cpu()*o_sy[node].cpu()]]

        mean = np.array(mean, dtype='float')
        cov = np.array(cov, dtype='float')
        next_values = np.random.multivariate_normal(mean, cov, 1)
        next_x[node] = next_values[0][0]
        next_y[node] = next_values[0][1]

    return next_x, next_y

def get_mean_error(ret_nodes, nodes, assumedNodesPresent, trueNodesPresent, using_cuda, look_up):
    '''
    Parameters
    ==========

    ret_nodes : A tensor of shape pred_length x numNodes x 2
    Contains the predicted positions for the nodes

    nodes : A tensor of shape pred_length x numNodes x 2gwt
    Contains the true positions for the nodes

    nodesPresent lists: A list of lists, of size pred_length
    Each list contains the nodeIDs of the nodes present at that time-step

    look_up : lookup table for determining which ped is in which array index

    Returns
    =======

    Error : Mean euclidean distance between predicted trajectory and the true trajectory
    '''
    pred_length = ret_nodes.size()[0]
    error = torch.zeros(pred_length)
    if using_cuda:
        error = error.cuda()

    for tstep in range(pred_length):
        counter = 0

        for nodeID in assumedNodesPresent[tstep]:
            nodeID = str(nodeID)

            if nodeID not in trueNodesPresent[tstep]:
                continue

            nodeID = look_up[nodeID]


            pred_pos = ret_nodes[tstep, nodeID, :]
            true_pos = nodes[tstep, nodeID, :]

            error[tstep] += torch.norm(pred_pos - true_pos, p=2)
            counter += 1

        if counter != 0:
            error[tstep] = error[tstep] / counter

    return torch.mean(error)


def get_final_error(ret_nodes, nodes, assumedNodesPresent, trueNodesPresent, look_up):
    '''
    Parameters
    ==========

    ret_nodes : A tensor of shape pred_length x numNodes x 2
    Contains the predicted positions for the nodes

    nodes : A tensor of shape pred_length x numNodes x 2
    Contains the true positions for the nodes

    nodesPresent lists: A list of lists, of size pred_length
    Each list contains the nodeIDs of the nodes present at that time-step

    look_up : lookup table for determining which ped is in which array index


    Returns
    =======

    Error : Mean final euclidean distance between predicted trajectory and the true trajectory
    '''
    pred_length = ret_nodes.size()[0]
    error = 0
    counter = 0

    # Last time-step
    tstep = pred_length - 1
    for nodeID in assumedNodesPresent[tstep]:
        nodeID = str(nodeID)


        if nodeID not in trueNodesPresent[tstep]:
            continue

        nodeID = look_up[nodeID]

        
        pred_pos = ret_nodes[tstep, nodeID, :]
        true_pos = nodes[tstep, nodeID, :]
        
        error += torch.norm(pred_pos - true_pos, p=2)
        counter += 1
        
    if counter != 0:
        error = error / counter
            
    return error


##################### Data related methods ######################


def sample_validation_data(x_seq, Pedlist, grid, args, net, look_up, num_pedlist, dataloader):
    '''
    The validation sample function
    params:
    x_seq: Input positions
    Pedlist: Peds present in each frame
    args: arguments
    net: The model
    num_pedlist : number of peds in each frame
    look_up : lookup table for determining which ped is in which array index
    '''
    # Number of peds in the sequence
    numx_seq = len(look_up)

    total_loss = 0

    # Construct variables for hidden and cell states
    with torch.no_grad():
        hidden_states = Variable(torch.zeros(numx_seq, net.args.rnn_size))
        if args.use_cuda:
            hidden_states = hidden_states.cuda()
        if not args.gru:
            cell_states = Variable(torch.zeros(numx_seq, net.args.rnn_size))
            if args.use_cuda:
                cell_states = cell_states.cuda()
        else:
            cell_states = None

        seq_length = x_seq.shape[0]
        ret_x_seq = Variable(torch.zeros(seq_length, numx_seq, 2))

        # Initialize the return data structure
        if args.use_cuda:
            ret_x_seq = ret_x_seq.cuda()

        ret_x_seq[0] = x_seq[0]

        if seq_length <= 1:
            return ret_x_seq, total_loss

        # For the observed part of the trajectory
        for tstep in range(seq_length - 1):
            loss = 0
            # Do a forward prop
            out_, hidden_states, cell_states = net(x_seq[tstep].view(1, numx_seq, 2), [grid[tstep]], hidden_states, cell_states, [Pedlist[tstep]], [num_pedlist[tstep]], dataloader, look_up)
            # loss_obs = Gaussian2DLikelihood(out_obs, x_seq[tstep+1].view(1, numx_seq, 2), [Pedlist[tstep+1]])

            # Extract the mean, std and corr of the bivariate Gaussian
            # mux, muy, sx, sy, corr = getCoef(out_)
            # Sample from the bivariate Gaussian
            next_x, next_y = out_[:, :, 0], out_[:, :, 1]
            ret_x_seq[tstep + 1, :, 0] = next_x
            ret_x_seq[tstep + 1, :, 1] = next_y
            #loss = Gaussian2DLikelihood(out_[0].view(1, out_.size()[1], out_.size()[2]), x_seq[tstep+1].view(1, numx_seq, 2), [Pedlist[tstep+1]], look_up)
            loss = RMSELoss(out_[0,:,:2].view(1, out_.size()[1], 2), x_seq[tstep+1].view(1, numx_seq, 2), [Pedlist[tstep+1]], look_up)
            total_loss += loss

    return ret_x_seq, total_loss / (seq_length-1)


def sample_validation_data_vanilla(x_seq, Pedlist, args, net, look_up, num_pedlist, dataloader):
    '''
    The validation sample function for vanilla method
    params:
    x_seq: Input positions
    Pedlist: Peds present in each frame
    args: arguments
    net: The model
    num_pedlist : number of peds in each frame
    look_up : lookup table for determining which ped is in which array index

    '''
    # Number of peds in the sequence
    numx_seq = len(look_up)

    total_loss = 0

    # Construct variables for hidden and cell states
    with torch.no_grad():
        hidden_states = Variable(torch.zeros(numx_seq, net.args.rnn_size))
        if args.use_cuda:
            hidden_states = hidden_states.cuda()
        if not args.gru:
            cell_states = Variable(torch.zeros(numx_seq, net.args.rnn_size))
            if args.use_cuda:
                cell_states = cell_states.cuda()
        else:
            cell_states = None

        seq_length = x_seq.shape[0]
        ret_x_seq = Variable(torch.zeros(seq_length, numx_seq, 2))

        # Initialize the return data structure
        if args.use_cuda:
            ret_x_seq = ret_x_seq.cuda()

        ret_x_seq[0] = x_seq[0]
        
        if seq_length <= 1:
            return ret_x_seq, total_loss

        # For the observed part of the trajectory
        for tstep in range(seq_length -1):
            loss = 0
            # Do a forward prop
            out_, hidden_states, cell_states = net(x_seq[tstep].view(1, numx_seq, 2), hidden_states, cell_states, [Pedlist[tstep]], [num_pedlist[tstep]], dataloader, look_up)
            # loss_obs = Gaussian2DLikelihood(out_obs, x_seq[tstep+1].view(1, numx_seq, 2), [Pedlist[tstep+1]])

            # Extract the mean, std and corr of the bivariate Gaussian
            #mux, muy, sx, sy, corr = getCoef(out_)
            # Sample from the bivariate Gaussian
            #next_x, next_y = sample_gaussian_2d(mux.data, muy.data, sx.data, sy.data, corr.data, Pedlist[tstep], look_up)
            next_x, next_y = out_[:, :, 0], out_[:, :, 1]
            ret_x_seq[tstep + 1, :, 0] = next_x
            ret_x_seq[tstep + 1, :, 1] = next_y
            #loss = Gaussian2DLikelihood(out_[0].view(1, out_.size()[1], out_.size()[2]), x_seq[tstep+1].view(1, numx_seq, 2), [Pedlist[tstep+1]], look_up)
            loss = RMSELoss(out_[0,:,:2].view(1, out_.size()[1], 2), x_seq[tstep+1].view(1, numx_seq, 2), [Pedlist[tstep+1]], look_up)
            total_loss += loss
            
    return ret_x_seq, total_loss / (seq_length-1)

def convert_proper_array(x_seq, num_pedlist, pedlist):
    '''
    converter function to appropriate format. Instead of directly use ship mmsi, we are mapping ship mmsi to
    array indices using a lookup table for each sequence -> speed
    output: [seq_length, numNode, 2] (lng, lat)
    [seq_length, (real sequence lenght+1)*max_ped_id+1 (biggest id number in the sequence)*2 (x,y)]
    '''
    #get unique ids from sequence
    unique_ids = pd.unique(np.concatenate(pedlist).ravel().tolist()).astype(str)
    # create a lookup table which maps ped ids -> array indices
    lookup_table = dict(zip(unique_ids, range(0, len(unique_ids))))

    seq_data = np.zeros(shape=(len(x_seq), len(lookup_table), 2))

    # create a lookup table which maps ped ids -> array indices
    for ind, frame in enumerate(x_seq):
        corr_index = [lookup_table[x] for x in frame[:, 0]]
        seq_data[ind, corr_index,:] = frame[:,1:3]

    return_arr = Variable(torch.from_numpy(np.array(seq_data)).float())

    return return_arr, lookup_table

def vectorize_seq(x_seq, PedsList_seq, lookup_seq):
    '''Substract first frame value to all frames for a ped.Therefore, convert absolute pos to relative pos.'''
    first_values_dict = WriteOnceDict()
    vectorized_x_seq = x_seq.clone()
    for ind, frame in enumerate(x_seq):
        for ped in PedsList_seq[ind]:
            first_values_dict[ped] = frame[lookup_seq[ped], 0:2]
            vectorized_x_seq[ind, lookup_seq[ped], 0:2]  = frame[lookup_seq[ped], 0:2] - first_values_dict[ped][0:2]

    return vectorized_x_seq, first_values_dict

def revert_seq(x_seq, PedsList_seq, lookup_seq, first_values_dict):
    '''Convert velocity array to absolute position array'''
    absolute_x_seq = x_seq.clone()
    for ind, frame in enumerate(x_seq):
        for ped in PedsList_seq[ind]:
            absolute_x_seq[ind, lookup_seq[ped], 0:2] = frame[lookup_seq[ped], 0:2] + first_values_dict[ped][0:2]

    return absolute_x_seq

def remove_file_extention(file_name):
    # remove file extension (.txt) given filename
    return file_name.split('.')[0]

def add_file_extention(file_name, extention):
    # add file extension (.txt) given filename

    return file_name + '.' + extention

def clear_folder(path):
    # remove all files in the folder
    if os.path.exists(path):
        shutil.rmtree(path)
        print("Folder succesfully removed: ", path)
    else:
        print("No such path: ",path)

def delete_file(path, file_name_list):
    # delete given file list
    for file in file_name_list:
        file_path = os.path.join(path, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print("File succesfully deleted: ", file_path)
            else:    ## Show an error ##
                print("Error: %s file not found" % file_path)        
        except OSError as e:  ## if failed, report it back to the user ##
            print ("Error: %s - %s." % (e.filename,e.strerror))

def get_all_file_names(path):
    # return all file names given directory
    files = []
    for (dirpath, dirnames, filenames) in walk(path):
        files.extend(filenames)
        break
    return files

def create_directories(base_folder_path, folder_list):
    # create folders using a folder list and path
    for folder_name in folder_list:
        directory = os.path.join(base_folder_path, folder_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

def unique_list(l):
  # get unique elements from list
  x = []
  for a in l:
    if a not in x:
      x.append(a)
  return x

def angle_between(p1, p2):
    # return angle between two points
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return ((ang1 - ang2) % (2 * np.pi))


def translate(x_seq, PedsList_seq, lookup_seq, value):
    # translate al trajectories given x and y values
    vectorized_x_seq = x_seq.clone()
    for ind, frame in enumerate(x_seq):
        for ped in PedsList_seq[ind]:
            vectorized_x_seq[ind, lookup_seq[ped], 0:2]  = frame[lookup_seq[ped], 0:2] - value[0:2]

    return vectorized_x_seq



def rotate(origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        #return torch.cat([qx, qy])
        return [qx, qy]

def time_lr_scheduler(optimizer, epoch, lr_decay=0.5, lr_decay_epoch=10):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch % lr_decay_epoch:
        return optimizer
    
    print("Optimizer learning rate has been decreased.")

    for param_group in optimizer.param_groups:
        param_group['lr'] *= (1. / (1. + lr_decay * epoch))
    return optimizer


def rotate_traj_with_target_ped(x_seq, angle, PedsList_seq, lookup_seq):
    # rotate sequence given angle
    origin = (0, 0)
    vectorized_x_seq = x_seq.clone()
    for ind, frame in enumerate(x_seq):
        for ped in PedsList_seq[ind]:
            point = frame[lookup_seq[ped], 0:2]
            rotated_point = rotate(origin, point, angle)
            vectorized_x_seq[ind, lookup_seq[ped], 0] = rotated_point[0]
            vectorized_x_seq[ind, lookup_seq[ped], 1] = rotated_point[1]
    return vectorized_x_seq