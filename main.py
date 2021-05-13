import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
import torch
from torch.utils.data import Dataset
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import csv
from sklearn.utils import shuffle

def read_file(path):
    df = pd.read_csv(path, sep=',', header=0, encoding='unicode_escape')
    return df

def mean_std(df):
    df=(df-df.mean()).div(df.std())
    return df

## normalize train
def normalize(csvfile0,csvfile1,outname,flag=0):
    train_df = read_file(csvfile0)
    train_t_df=read_file(csvfile1)
    t_values=train_t_df.loc[:,'value_t-1':'value_t-7']
    train_df['label'] = train_df['country'].rank(method='dense', ascending=True).astype(int)
    country_label=train_df['label']
    if(flag==0):
        label = train_df[['next_week_hospitalizations']]
        feat = train_df.drop(['country','next_week_hospitalizations', 'date', 'year_week'], axis=1)
    else:
        feat = train_df.drop(['country', 'date', 'year_week'], axis=1)

    feat = pd.concat([feat, t_values], axis=1)
    dfs=[]
    for i in range(0,15):
        df=mean_std(feat[feat['label']==(i+1)])
        dfs.append(df)
    for i in range(1,15):
        dfs[0]=pd.concat([dfs[0],dfs[i]])
    feat=dfs[0]
    feat = feat.drop(['label'], axis=1)

    feat=pd.concat([feat,country_label],axis=1)
    if(flag==0):
        feat=pd.concat([feat,label],axis=1)
    feat.to_csv(outname,index=False)

normalize("train_creative.csv","train_creative_t_values.csv","train.csv")
normalize("test_creative_no_label.csv","test_creative_t_values.csv","test.csv",1)

def split(train_df):
    labels= train_df[['next_week_hospitalizations']]
    country_labels=train_df[['label']]
    feat= train_df.drop(['next_week_hospitalizations','label'],axis=1)
    x_train, x_test, y_train, y_test,z_train,z_test = train_test_split(feat, labels,country_labels, test_size=0.1, random_state=42)
    x_train=np.array(x_train)
    x_test=np.array(x_test)
    y_train=np.array(y_train)
    y_test=np.array(y_test)
    z_train=np.array(z_train)
    z_test=np.array(z_test)
    return x_train,x_test,y_train,y_test,z_train,z_test

def non_split(train_df):
    labels= train_df[['next_week_hospitalizations']]
    country_labels=train_df[['label']]
    feat= train_df.drop(['next_week_hospitalizations','label'],axis=1)
    x=np.array(feat)
    y=np.array(labels)
    z=np.array(country_labels)
    return x,y,z

class CSVDataset(Dataset):
    def __init__(self, train_df,label_df,country_labels,flag=0):
        # Where the initial logic happens like reading a csv, doing data augmentation, etc.
        self.length=len(train_df)
        if(flag==0):
            self.country_labels=shuffle(country_labels,random_state=42)
            self.labels = shuffle(label_df,random_state=42)
            self.feat = shuffle(train_df,random_state=42)
        else:
            self.country_labels=country_labels
            self.labels = label_df
            self.feat = train_df

    def __len__(self):
        # Returns count of samples (an integer) you have.
        return self.length

    def __getitem__(self, idx):
        # Given an index, returns the correponding datapoint.
        # This function is called from dataloader like this:
        # img, label = CSVDataset.__getitem__(99)  # For 99th item
        return self.feat[idx],self.labels[idx],self.country_labels[idx]

class Feedforward(torch.nn.Module):
    def __init__(self):
        super(Feedforward, self).__init__()
        self.fc1 = torch.nn.Linear(14, 7)
        #self.fc2 = torch.nn.Linear(14, 7)
        self.fc3 = torch.nn.Linear(7, 1)
        self.relu = torch.nn.ReLU()
        self.sig=nn.Sigmoid()

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        #h2 = self.relu(self.fc2(h1))
        output=self.fc3(h1)
        return output


def train(csvfile,csvfile1,flag=0):
    train_df=read_file(csvfile)
    test_df=read_file(csvfile1)
    nets = []
    criterions = []
    optimizers = []
    for i in range(0, 15):
        # Initialize an object of the model class
        net = Feedforward()
        nets.append(net)
        # Define your loss function
        criterion = nn.MSELoss()
        criterions.append(criterion)
        # Create your optimizer
        optimizer = optim.SGD(net.parameters(), momentum=0.9, weight_decay=0.15, lr=1 * 10 ** -7)
        optimizers.append(optimizer)
    if(flag==0):
        x_train, x_test, y_train, y_test,z_train,z_test = split(train_df)
          # Initialize an object of the dataset class
        dataset = CSVDataset(x_train,y_train,z_train)
        dataset2 = CSVDataset(x_test,y_test,z_test)
        # Wrap a dataloader around the dataset object.
        dataloader = DataLoader(dataset)
        dataloader2 = DataLoader(dataset2)
        # Beging training!
        for epoch in range(20):
            running_loss = 0.0
            for i, (input, target,country_label) in enumerate(dataloader):
                # You always want to use zero_grad(), backward(), and step() in the following order.
                # zero_grad clears old gradients from the last step (otherwise you’d just accumulate the gradients from all loss.backward() calls).
                optimizers[country_label-1].zero_grad()
                # As said before, you can only code as below if your network belongs to the nn.Module class.
                output = nets[country_label-1](torch.tensor(input,dtype=torch.float))
                loss = criterions[country_label-1](output, torch.tensor(target,dtype=(torch.float)))
                # loss.backward() computes the derivative of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation.
                loss.backward()
                # optimizer.step() causes the optimizer to take a step based on the gradients of the parameters.
                optimizers[country_label-1].step()
                running_loss += loss.item()
                if i % 1000 == 999:  # print every 2000 mini-batches
                    #print('[%d, %5d] training_loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
                    running_loss = 0.0
            print("end of epoch {}".format(epoch+1))
            running_loss4 = 0.0
            for i, (input, target,country_label) in enumerate(dataloader2):
                output = nets[country_label-1](torch.tensor(input,dtype=torch.float))
                #print(output)
                loss = criterions[country_label-1](output, torch.tensor(target,dtype=(torch.float)))
                running_loss4 += loss.item()
            print('valid_loss: %.3f' % (running_loss4 / 400))
    else:
        x,y,z=non_split(train_df)
        dataset = CSVDataset(x, y, z)
        dataloader = DataLoader(dataset)
        for epoch in range(20):
            for i, (input, target,country_label) in enumerate(dataloader):
                # You always want to use zero_grad(), backward(), and step() in the following order.
                # zero_grad clears old gradients from the last step (otherwise you’d just accumulate the gradients from all loss.backward() calls).
                optimizers[country_label-1].zero_grad()
                # As said before, you can only code as below if your network belongs to the nn.Module class.
                output = nets[country_label-1](torch.tensor(input,dtype=torch.float))
                loss = criterions[country_label-1](output, torch.tensor(target,dtype=(torch.float)))
                # loss.backward() computes the derivative of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation.
                loss.backward()
                # optimizer.step() causes the optimizer to take a step based on the gradients of the parameters.
                optimizers[country_label-1].step()
            print("end of epoch {}".format(epoch+1))
        pred=[]
        pred_z=np.array(test_df[['label']])
        pred_x=np.array(test_df.drop(['label'],axis=1))
        dataset2 = CSVDataset(pred_x, y, pred_z,1)
        dataloader2 = DataLoader(dataset2)

        for i, (input, target, country_label) in enumerate(dataloader2):
            output = nets[country_label - 1](torch.tensor(input, dtype=torch.float))
            pred.append(output.data.item())
        test_df = read_file("test_creative_no_label.csv")
        # categorical_columns = ['country']
        combine_lambda = lambda x: '{} {}'.format(x.country, x.date)
        submit_df = pd.DataFrame({'country_id' : [], 'next_week_hospitalizations': []})
        submit_df['country_id'] = test_df.apply(combine_lambda, axis=1)
        submit_df['next_week_hospitalizations'] = pred
        submit_df.to_csv('out.csv',index=False)


train("train.csv","test.csv",1)
#y_pred = Feedforward("valiadation.csv")
#print(mean_squared_error(y_test, y_pred))