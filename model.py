import torch
import torch.nn as nn
import numpy as np
import itertools

from torch.autograd import Variable

class SocialModel(nn.Module):

    def __init__(self, args, infer=False):
        '''
        Initializer function
        params:
        args: Training arguments
        infer: Training or test time (true if test time)
        '''
        super(SocialModel, self).__init__()

        self.args = args
        self.infer = infer
        self.use_cuda = args.use_cuda

        if infer:
            # Test time
            self.seq_length = 1
        else:
            # Training time
            self.seq_length = args.seq_length

        # Store required sizes
        self.rnn_size = args.rnn_size
        self.grid_size = args.grid_size
        self.embedding_size = args.embedding_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.maxNumPeds=args.maxNumPeds
        self.seq_length=args.seq_length
        self.gru = args.gru


        # The LSTM cell
        self.cell = nn.LSTMCell(2*self.embedding_size, self.rnn_size)

        if self.gru:
            self.cell = nn.GRUCell(2*self.embedding_size, self.rnn_size)


        # Linear layer to embed the input position
        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)
        # Linear layer to embed the social tensor
        self.tensor_embedding_layer = nn.Linear(self.grid_size*self.grid_size*self.rnn_size, self.embedding_size)

        # Linear layer to map the hidden state of LSTM to output
        self.output_layer = nn.Linear(self.rnn_size, self.output_size)

        # ReLU and dropout unit
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

    def getSocialTensor(self, grid, hidden_states):
        '''
        Computes the social tensor for a given grid mask and hidden states of all peds
        params:
        grid : Grid masks
        hidden_states : Hidden states of all peds
        '''
        # Number of peds
        numNodes = grid.size()[0]

        # Construct the variable
        social_tensor = Variable(torch.zeros(numNodes, self.grid_size*self.grid_size, self.rnn_size))
        if self.use_cuda:
            social_tensor = social_tensor.cuda()
        
        # For each ped
        for node in range(numNodes):
            # Compute the social tensor
            social_tensor[node] = torch.mm(torch.t(grid[node]), hidden_states)

        # Reshape the social tensor
        social_tensor = social_tensor.view(numNodes, self.grid_size*self.grid_size*self.rnn_size)
        return social_tensor
            
    #def forward(self, input_data, grids, hidden_states, cell_states ,PedsList, num_pedlist,dataloader, look_up):
    def forward(self, *args):

        '''
        Forward pass for the model
        params:
        input_data: Input positions
        grids: Grid masks
        hidden_states: Hidden states of the peds
        cell_states: Cell states of the peds
        PedsList: id of peds in each frame for this sequence

        returns:
        outputs_return: Outputs corresponding to bivariate Gaussian distributions
        hidden_states
        cell_states
        '''
        # Construct the output variable
        input_data = args[0]
        grids = args[1]
        hidden_states = args[2]
        cell_states = args[3]

        if self.gru:
            cell_states = None

        PedsList = args[4]
        num_pedlist = args[5]
        dataloader = args[6]
        look_up = args[7]
        #if not self.infer:
        self.seq_length = len(PedsList)

        numNodes = len(look_up)
        outputs = Variable(torch.zeros(self.seq_length * numNodes, self.output_size))
        if self.use_cuda:            
            outputs = outputs.cuda()

        # For each frame in the sequence
        for framenum, frame in enumerate(input_data):

            # Peds present in the current frame
            nodeIDs = [str(nodeID) for nodeID in PedsList[framenum]]

            if len(nodeIDs) == 0:
                # If no peds, then go to the next frame
                continue

            # List of nodes
            list_of_nodes = [look_up[x] for x in nodeIDs]

            corr_index = Variable((torch.LongTensor(list_of_nodes)))
            if self.use_cuda:            
                corr_index = corr_index.cuda()

            # Select the corresponding input positions
            nodes_current = frame[list_of_nodes,:]
            # Get the corresponding grid masks
            grid_current = grids[framenum]

            # Get the corresponding hidden and cell states
            hidden_states_before = torch.index_select(hidden_states, 0, corr_index)

            if not self.gru:
                cell_states_before = torch.index_select(cell_states, 0, corr_index)

            # Compute the social tensor
            social_tensor = self.getSocialTensor(grid_current, hidden_states_before)

            # Embed inputs
            input_embedded = self.dropout(self.relu(self.input_embedding_layer(nodes_current)))
            # Embed the social tensor
            tensor_embedded = self.dropout(self.relu(self.tensor_embedding_layer(social_tensor)))

            # Concat input
            concat_embedded = torch.cat((input_embedded, tensor_embedded), 1)

            if not self.gru:
                # One-step of the LSTM
                hidden_states_current, cell_states_current = self.cell(concat_embedded, (hidden_states_before, cell_states_before))
            else:
                hidden_states_current = self.cell(concat_embedded, (hidden_states_before))

            # Compute the output
            outputs[framenum*numNodes + corr_index.data] = self.output_layer(hidden_states_current)

            # Update hidden and cell states
            hidden_states[corr_index.data] = hidden_states_current
            if not self.gru:
                cell_states[corr_index.data] = cell_states_current

        # Reshape outputs
        outputs_return = Variable(torch.zeros(self.seq_length, numNodes, self.output_size))
        if self.use_cuda:
            outputs_return = outputs_return.cuda()
        for framenum in range(self.seq_length):
            for node in range(numNodes):
                outputs_return[framenum, node, :] = outputs[framenum*numNodes + node, :]

        return outputs_return, hidden_states, cell_states


class VLSTMModel(nn.Module):

    def __init__(self, args, infer=False):
        '''
        Initializer function
        params:
        args: Training arguments
        infer: Training or test time (true if test time)
        '''
        super(VLSTMModel, self).__init__()

        self.args = args
        self.infer = infer
        self.use_cuda = args.use_cuda

        if infer:
            # Test time
            self.seq_length = 1
        else:
            # Training time
            self.seq_length = args.seq_length

        # Store required sizes
        self.rnn_size = args.rnn_size
        self.embedding_size = args.embedding_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.maxNumPeds=args.maxNumPeds
        self.seq_length=args.seq_length
        self.gru = args.gru


        # The LSTM cell
        self.cell = nn.LSTMCell(self.embedding_size, self.rnn_size)

        if self.gru:
            self.cell = nn.GRUCell(self.embedding_size, self.rnn_size)


        # Linear layer to embed the input position
        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)
        # Linear layer to embed the social tensor
        #self.tensor_embedding_layer = nn.Linear(self.grid_size*self.grid_size, self.embedding_size)

        # Linear layer to map the hidden state of LSTM to output
        self.output_layer = nn.Linear(self.rnn_size, self.output_size)

        # ReLU and dropout unit
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)
            
    #def forward(self, input_data, grids, hidden_states, cell_states ,PedsList, num_pedlist,dataloader, look_up):
    def forward(self, *args):

        '''
        Forward pass for the model
        params:
        input_data: Input positions
        grids: Grid masks
        hidden_states: Hidden states of the peds
        cell_states: Cell states of the peds
        PedsList: id of peds in each frame for this sequence

        returns:
        outputs_return: Outputs corresponding to bivariate Gaussian distributions
        hidden_states
        cell_states
        '''
        # List of tensors each of shape args.maxNumPedsx3 corresponding to each frame in the sequence
            # frame_data = tf.split(0, args.seq_length, self.input_data, name="frame_data")
        #frame_data = [torch.squeeze(input_, [0]) for input_ in torch.split(0, self.seq_length, input_data)]
        
        #print("***************************")
        #print("input data")
        # Construct the output variable
        input_data = args[0]
        hidden_states = args[1]
        cell_states = args[2]

        if self.gru:
            cell_states = None

        PedsList = args[3]
        num_pedlist = args[4]
        dataloader = args[5]
        look_up = args[6]
        #if not self.infer:
        self.seq_length = len(PedsList)

        numNodes = len(look_up)
        outputs = Variable(torch.zeros(self.seq_length * numNodes, self.output_size))
        if self.use_cuda:            
            outputs = outputs.cuda()

        # For each frame in the sequence
        for framenum, frame in enumerate(input_data):

            # Peds present in the current frame

            #print("now processing: %s base frame number: %s, in-frame: %s"%(dataloader.get_test_file_name(), dataloader.frame_pointer, framenum))
            #print("list of nodes")

            nodeIDs_boundary = num_pedlist[framenum]
            nodeIDs = [str(nodeID) for nodeID in PedsList[framenum]]
            #print(PedsList)

            if len(nodeIDs) == 0:
                # If no peds, then go to the next frame
                continue


            # List of nodes
            #print("lookup table :%s"% look_up)
            list_of_nodes = [look_up[x] for x in nodeIDs]

            corr_index = Variable((torch.LongTensor(list_of_nodes)))
            if self.use_cuda:            
                outputs = outputs.cuda()
                corr_index = corr_index.cuda()
            #print("list of nodes: %s"%nodeIDs)
            #print("trans: %s"%corr_index)
            #if self.use_cuda:
             #   list_of_nodes = list_of_nodes.cuda()


            #print(list_of_nodes.data)
            # Select the corresponding input positions
            nodes_current = frame[list_of_nodes,:]
            # Get the corresponding grid masks


            # Get the corresponding hidden and cell states
            hidden_states_before = torch.index_select(hidden_states, 0, corr_index)

            if not self.gru:
                cell_states_before = torch.index_select(cell_states, 0, corr_index)


            # Embed inputs
            input_embedded = self.dropout(self.relu(self.input_embedding_layer(nodes_current)))

            if not self.gru:
                # One-step of the LSTM
                hidden_states_current, cell_states_current = self.cell(input_embedded, (hidden_states_before, cell_states_before))
            else:
                hidden_states_current = self.cell(input_embedded, (hidden_states_before))


            # Compute the output
            outputs[framenum*numNodes + corr_index.data] = self.output_layer(hidden_states_current)

            # Update hidden and cell states
            hidden_states[corr_index.data] = hidden_states_current
            if not self.gru:
                cell_states[corr_index.data] = cell_states_current

        # Reshape outputs
        outputs_return = Variable(torch.zeros(self.seq_length, numNodes, self.output_size))
        if self.use_cuda:
            outputs_return = outputs_return.cuda()
        for framenum in range(self.seq_length):
            for node in range(numNodes):
                outputs_return[framenum, node, :] = outputs[framenum*numNodes + node, :]

        return outputs_return, hidden_states, cell_states


class OLSTMModel(nn.Module):

    def __init__(self, args, infer=False):
        '''
        Initializer function
        params:
        args: Training arguments
        infer: Training or test time (true if test time)
        '''
        super(OLSTMModel, self).__init__()

        self.args = args
        self.infer = infer
        self.use_cuda = args.use_cuda

        if infer:
            # Test time
            self.seq_length = 1
        else:
            # Training time
            self.seq_length = args.seq_length

        # Store required sizes
        self.rnn_size = args.rnn_size
        self.grid_size = args.grid_size
        self.embedding_size = args.embedding_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.maxNumPeds=args.maxNumPeds
        self.seq_length=args.seq_length
        self.gru = args.gru


        # The LSTM cell
        self.cell = nn.LSTMCell(2*self.embedding_size, self.rnn_size)

        if self.gru:
            self.cell = nn.GRUCell(2*self.embedding_size, self.rnn_size)


        # Linear layer to embed the input position
        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)
        # Linear layer to embed the social tensor
        self.tensor_embedding_layer = nn.Linear(self.grid_size*self.grid_size, self.embedding_size)

        # Linear layer to map the hidden state of LSTM to output
        self.output_layer = nn.Linear(self.rnn_size, self.output_size)

        # ReLU and dropout unit
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

    def getObsTensor(self, grid):
        '''
        Computes the obstacle map tensor for a given grid mask and hidden states of all peds
        params:
        grid : Grid masks
        '''
        # Number of peds
        numNodes = grid.size()[0]
        # Construct the variable
        Obs_tensor = Variable(torch.zeros(numNodes, self.grid_size*self.grid_size))
        if self.use_cuda:
            Obs_tensor = Obs_tensor.cuda()
        # For each ped
        for node in range(numNodes):
            # Compute the obstacle tensor
            Obs_tensor[node] = (grid[node])

        # Reshape the social tensor
        Obs_tensor = Obs_tensor.view(numNodes, self.grid_size*self.grid_size)
        return Obs_tensor
            
    #def forward(self, input_data, grids, hidden_states, cell_states ,PedsList, num_pedlist,dataloader, look_up):
    def forward(self, *args):

        '''
        Forward pass for the model
        params:
        input_data: Input positions
        grids: Grid masks
        hidden_states: Hidden states of the peds
        cell_states: Cell states of the peds
        PedsList: id of peds in each frame for this sequence

        returns:
        outputs_return: Outputs corresponding to bivariate Gaussian distributions
        hidden_states
        cell_states
        '''
        # List of tensors each of shape args.maxNumPedsx3 corresponding to each frame in the sequence
            # frame_data = tf.split(0, args.seq_length, self.input_data, name="frame_data")
        #frame_data = [torch.squeeze(input_, [0]) for input_ in torch.split(0, self.seq_length, input_data)]
        
        #print("***************************")
        #print("input data")
        # Construct the output variable
        input_data = args[0]
        grids = args[1]
        hidden_states = args[2]
        cell_states = args[3]

        if self.gru:
            cell_states = None

        PedsList = args[4]
        num_pedlist = args[5]
        dataloader = args[6]
        look_up = args[7]
        if not self.infer:
            self.seq_length = len(PedsList)
            
        numNodes = len(look_up)
        outputs = Variable(torch.zeros(self.seq_length * numNodes, self.output_size))
        if self.use_cuda:            
            outputs = outputs.cuda()

        # For each frame in the sequence
        for framenum,frame in enumerate(input_data):

            # Peds present in the current frame

            #print("now processing: %s base frame number: %s, in-frame: %s"%(dataloader.get_test_file_name(), dataloader.frame_pointer, framenum))
            #print("list of nodes")

            nodeIDs_boundary = num_pedlist[framenum]
            nodeIDs = [str(nodeID) for nodeID in PedsList[framenum]]
            #print(PedsList)

            if len(nodeIDs) == 0:
                # If no peds, then go to the next frame
                continue


            # List of nodes
            #print("lookup table :%s"% look_up)
            list_of_nodes = [look_up[x] for x in nodeIDs]

            corr_index = Variable((torch.LongTensor(list_of_nodes)))
            if self.use_cuda:
                corr_index = corr_index.cuda()
            #print("list of nodes: %s"%nodeIDs)
            #print("trans: %s"%corr_index)
            #if self.use_cuda:
             #   list_of_nodes = list_of_nodes.cuda()


            #print(list_of_nodes.data)
            # Select the corresponding input positions
            nodes_current = frame[list_of_nodes,:]
            # Get the corresponding grid masks
            grid_current = grids[framenum]

            # Get the corresponding hidden and cell states
            hidden_states_before = torch.index_select(hidden_states, 0, corr_index)

            if not self.gru:
                cell_states_before = torch.index_select(cell_states, 0, corr_index)

            #print(grid_current.shape)
            #print(hidden_states_before.shape)
            # Compute the social tensor
            Obs_tensor = self.getObsTensor(grid_current)

            # Embed inputs
            input_embedded = self.dropout(self.relu(self.input_embedding_layer(nodes_current)))
            # Embed the social tensor
            tensor_embedded = self.dropout(self.relu(self.tensor_embedding_layer(Obs_tensor)))

            # Concat input
            concat_embedded = torch.cat((input_embedded, tensor_embedded), 1)

            if not self.gru:
                # One-step of the LSTM
                hidden_states_current, cell_states_current = self.cell(concat_embedded, (hidden_states_before, cell_states_before))
            else:
                hidden_states_current = self.cell(concat_embedded, (hidden_states_before))


            # Compute the output
            outputs[framenum*numNodes + corr_index.data] = self.output_layer(hidden_states_current)

            # Update hidden and cell states
            hidden_states[corr_index.data] = hidden_states_current
            if not self.gru:
                cell_states[corr_index.data] = cell_states_current

        # Reshape outputs
        outputs_return = Variable(torch.zeros(self.seq_length, numNodes, self.output_size))
        if self.use_cuda:
            outputs_return = outputs_return.cuda()
        for framenum in range(self.seq_length):
            for node in range(numNodes):
                outputs_return[framenum, node, :] = outputs[framenum*numNodes + node, :]

        return outputs_return, hidden_states, cell_states


def getGridMask(frame, num_person, neighborhood_size, grid_size, is_occupancy = False):
    '''
    This function computes the binary mask that represents the
    occupancy of each ped in the other's grid
    params:
    frame : This will be a MNP x 3 matrix with each row being [shipID, x, y]
    neighborhood_size : Scalar value representing the size of neighborhood considered
    grid_size : Scalar value representing the size of the grid discretization
    num_person : number of people exist in given frame
    is_occupancy: A flag using for calculation of accupancy map

    '''
    mnp = num_person

    if is_occupancy:
        frame_mask = np.zeros((mnp, grid_size**2))
    else:
        frame_mask = np.zeros((mnp, mnp, grid_size**2))
    frame_np =  frame.data.numpy()

    #width_bound, height_bound = (neighborhood_size/(width*1.0)), (neighborhood_size/(height*1.0))
    width_bound, height_bound = neighborhood_size, neighborhood_size
    #print("weight_bound: ", width_bound, "height_bound: ", height_bound)

    #instead of 2 inner loop, we check all possible 2-permutations which is 2 times faster.
    list_indices = list(range(0, mnp))
    for real_frame_index, other_real_frame_index in itertools.permutations(list_indices, 2):
        current_x, current_y = frame_np[real_frame_index, 0], frame_np[real_frame_index, 1]

        width_low, width_high = current_x - width_bound/2, current_x + width_bound/2
        height_low, height_high = current_y - height_bound/2, current_y + height_bound/2

        other_x, other_y = frame_np[other_real_frame_index, 0], frame_np[other_real_frame_index, 1]
        
        #if (other_x >= width_high).all() or (other_x < width_low).all() or (other_y >= height_high).all() or (other_y < height_low).all():
        if (other_x >= width_high) or (other_x < width_low) or (other_y >= height_high) or (other_y < height_low):
                # Ped not in surrounding, so binary mask should be zero
                #print("not surrounding")
                continue
        # If in surrounding, calculate the grid cell
        cell_x = int(np.floor(((other_x - width_low)/width_bound) * grid_size))
        cell_y = int(np.floor(((other_y - height_low)/height_bound) * grid_size))

        if cell_x >= grid_size or cell_x < 0 or cell_y >= grid_size or cell_y < 0:
                continue

        if is_occupancy:
            frame_mask[real_frame_index, cell_x + cell_y*grid_size] = 1
        else:
            # Other ped is in the corresponding grid cell of current ped
            frame_mask[real_frame_index, other_real_frame_index, cell_x + cell_y*grid_size] = 1

    #Two inner loops aproach -> slower
    # # For each ped in the frame (existent and non-existent)
    # for real_frame_index in range(mnp):
    #     #real_frame_index = lookup_seq[pedindex]
    #     #print(real_frame_index)
    #     #print("****************************************")
    #     # Get x and y of the current ped
    #     current_x, current_y = frame[real_frame_index, 0], frame[real_frame_index, 1]

    #     #print("cur x : ", current_x, "cur_y: ", current_y)

    #     width_low, width_high = current_x - width_bound/2, current_x + width_bound/2
    #     height_low, height_high = current_y - height_bound/2, current_y + height_bound/2
    #     #print("width_low : ", width_low, "width_high: ", width_high, "height_low : ", height_low, "height_high: ", height_high)


    #     # For all the other peds
    #     for other_real_frame_index in range(mnp):
    #         #other_real_frame_index = lookup_seq[otherpedindex]
    #         #print(other_real_frame_index)


    #         #print("################################")
    #         # If the other pedID is the same as current pedID
    #         if other_real_frame_index == real_frame_index:
    #             # The ped cannot be counted in his own grid
    #             continue

    #         # Get x and y of the other ped
    #         other_x, other_y = frame[other_real_frame_index, 0], frame[other_real_frame_index, 1]
    #         #print("other_x: ", other_x, "other_y: ", other_y)
    #         if (other_x >= width_high).all() or (other_x < width_low).all() or (other_y >= height_high).all() or (other_y < height_low).all():
    #             # Ped not in surrounding, so binary mask should be zero
    #             #print("not surrounding")
    #             continue

    #         # If in surrounding, calculate the grid cell
    #         cell_x = int(np.floor(((other_x - width_low)/width_bound) * grid_size))
    #         cell_y = int(np.floor(((other_y - height_low)/height_bound) * grid_size))
    #         #print("cell_x: ", cell_x, "cell_y: ", cell_y)

    #         if cell_x >= grid_size or cell_x < 0 or cell_y >= grid_size or cell_y < 0:
    #             continue

    #         # Other ped is in the corresponding grid cell of current ped
    #         frame_mask[real_frame_index, other_real_frame_index, cell_x + cell_y*grid_size] = 1
    # #print("frame mask shape %s"%str(frame_mask.shape))

    return frame_mask

def getSequenceGridMask(sequence, shipslist_seq, neighborhood_size, grid_size, using_cuda, is_occupancy=False):
    '''
    Get the grid masks for all the frames in the sequence
    params:
    sequence : A numpy matrix of shape SL x MNP x 3
    neighborhood_size : Scalar value representing the size of neighborhood considered
    grid_size : Scalar value representing the size of the grid discretization
    using_cuda: Boolean value denoting if using GPU or not
    is_occupancy: A flag using for calculation of accupancy map
    '''
    sl = len(sequence)
    sequence_mask = []

    for i in range(sl):
        mask = Variable(torch.from_numpy(getGridMask(sequence[i], len(shipslist_seq[i]), neighborhood_size, grid_size, is_occupancy)).float())
        if using_cuda:
            mask = mask.cuda()
        sequence_mask.append(mask)

    return sequence_mask
