
import torch
from torch.utils.data import DataLoader
import torch.nn as nn


from Task1.network import TemporalConvNet
import numpy as np

from Task1.dataLoader import datasetCSV
import torch.utils.tensorboard


#!/usr/bin/env python
config = {
"learning_rate" : 1e-3,
"num_epochs" : 60,
"decay" : 1e-5,
"input_dim" : 24,
"seq_dim" : 512,
"batch_size" :128,
"num_workers":4
}

# Create Sequence(input_feature_window and output window for currect Univariate Time series)
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = np.array(input_data).shape[1] # no of columns/sequence
    for i in range(L-tw-1):
        train_seq = input_data[:, i:i+tw]
        output = input_data[:, i + 1:i + tw + 1]
        inout_seq.append((train_seq, output))
    return inout_seq


def train(npy_file_path):

    # load dataset
    df = np.load(npy_file_path)
    rmse_list = []

    # For each time series, train; save model and calculate rmse
    for time_series_id in range(228):
        model = TemporalConvNet(
            num_inputs=1,
            num_channels=[32, 32, 32, 32, 32, 1],
            kernel_size=7,
            dropout=0.3,
            init=True,
        )

        if torch.cuda.is_available():
            model.cuda()
        print('cuda available: ', torch.cuda.is_available())

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

        criterion = nn.L1Loss()


        # generate train sequence list based upon above dataframe.
        trainDataseq = create_inout_sequences(df[time_series_id: time_series_id + 1, :-2880], config['seq_dim'])
        trainDataset = datasetCSV(trainDataseq, config['seq_dim'])
        trainLoader = DataLoader(trainDataset, batch_size=config['batch_size'], shuffle=False, drop_last=True)

        valDataseq = create_inout_sequences(df[time_series_id: time_series_id + 1:, -2880: -1440], config['seq_dim'])
        valDataset = datasetCSV(valDataseq, config['seq_dim'])
        valLoader = DataLoader(valDataset, batch_size=config['batch_size'], shuffle=False, drop_last=True)

        # testDataseq = create_inout_sequences(df[time_series_id: time_series_id + 1, : -1440-12], config['seq_dim'])
        # testDataset = datasetCSV(testDataseq, config['seq_dim'])
        # testLoader = DataLoader(testDataset, batch_size=config['batch_size'], shuffle=False, drop_last=True)

        training(config['num_epochs'],  trainLoader, optimizer, model, criterion, config['seq_dim'],
                 config['input_dim'], config['batch_size'],
                 df, valLoader)

        model_path ='./models/' + str(time_series_id)
        torch.save(model, model_path)

        # Taking Last 1440+12 values to predict 1440 values in rolling window fashion
        rmse = predict_future(df[time_series_id: time_series_id + 1, -1440-12:], model)
        rmse_list.append(rmse)
    print('Mean RMSE is: ',np.mean(rmse_list))


# Train the Network
def training(num_epochs, trainLoader, optimizer, model, criterion, seq_dim, input_dim, batch_size, df, valLoader):

    print('Training Started')
    for epoch in range(num_epochs):
        print('epoch is', epoch)
        running_loss = 0
        for i, (input, output) in enumerate(trainLoader):
            if i % 10 == 0:
                print(i)
            if torch.cuda.is_available():
                input = input.float().cuda()
                output = output.float().cuda()
            else:
                input = input.float()
                output = output.float()

            # Network is taking  input size as (None,1,512)
            # input = torch.squeeze(input,dim=0)
            # output = torch.squeeze(output, dim=0)

            optimizer.zero_grad()  # Reset gradients tensors

            predicted = model(input)

            loss = criterion(predicted, output) / torch.abs(output.data).mean()

            running_loss += loss
            loss.backward()  # Backward pass
            optimizer.step()  # Now we can do an optimizer step

        if epoch % 5 == 0:
            trainLoss = evaluate(trainLoader, model, config['seq_dim'], config['input_dim'], config['batch_size'], criterion, optimizer)
            print('epoch is: ', epoch)
            print('trainLoss is {}.'.format(trainLoss))

            # Loggin trainloss
            valLoss = evaluate(valLoader, model, config['seq_dim'], config['input_dim'],
             config['batch_size'], criterion, optimizer)

            print('valLoss is {}. '.format(valLoss))
            print('\n\n --------------------------------------------------------------------------------------------\n\n')

# Evaluate the network and return the loss
def evaluate(loader, model, seq_dim, input_dim, batch_size, criterion, optimizer):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    running_loss = 0
    with torch.no_grad():
        for i, (input, output) in enumerate(loader):

            if torch.cuda.is_available():
                input = input.float().cuda()
                output = output.float().cuda()
            else:
                input = input.float()
                output = output.float()

            input = torch.squeeze(input,dim=0)
            output = torch.squeeze(output, dim=0)

            predicted = model(input)

            loss = criterion(predicted, output) / torch.abs(output.data).mean()

            running_loss += loss

    return  running_loss

# give next 9 future values based pn 12 last values, in rolling window style
def predict_future_batch(seq,
    data, future=9, cpu=True
):
    if cpu:
        seq = seq.cpu()

    out = seq(data)
    ci = data.size(2)
    output = out[:, :, out.size(2) - 1].view(out.size(0), out.size(1), 1)

    out = torch.cat((data, output), dim=2)
    for i in range(future - 1):
        inp = out
        out = seq(inp)
        output = out[:, :, out.size(2) - 1].view(out.size(0), out.size(1), 1)
        ci += 1

        out = torch.cat((inp, output), dim=2)
    out = out[:, 0, :].view(out.size(0), 1, out.size(2))
    return out

def predict_future(
    data,
    seq,
    future=9,
    cpu=False,
    bsize=1,
):
    actual = data
    actual = np.expand_dims(actual, axis=1)
    actual = torch.from_numpy(actual).float()

    data = np.expand_dims(data, axis=1)
    data = torch.from_numpy(data).float()
    predicted = torch.zeros(1,1,0)

    # Predict until we get 1440 values
    while(predicted.size()[2] <1440):
        out = predict_future_batch(seq, data[:, :, -12:])
        out = out[:,:,-9:]
        data = torch.cat((data, out), dim=-1)
        predicted = torch.cat((predicted,out), dim=-1)

    actual = actual[:,:,-1440:]
    rmse = np.sqrt(((predicted.detach().numpy() - actual[:,:,-1440:].detach().numpy()) ** 2).mean())
    print(rmse)
    return rmse


if __name__ == "__main__":
    npy_file_path = '../../datasets/pems.npy'
    train(npy_file_path)
