import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

dir_path = os.getcwd()
train_baseline_file = os.path.join(dir_path, "train1.csv")
test_baseline_no_label = os.path.join(dir_path, "test_creative_no_label.csv")

def read_file(path):
    df = pd.read_csv(path, sep=',', header=0, encoding='unicode_escape')
    return df

def transform_label(labels):
    out_df=[]
    out=[]
    for i in range(0,10):
        tmp=[0]*10
        tmp[i]=1
        out.append(tmp)
    for j in range(0,len(labels)):
        data=labels.loc[j].values[0]
        out_df.append(out[data])
    return out_df

##model solution
##handle label
# train_df = read_file(train_baseline_file)
# labels_g = train_df[['next_week_hospitalizations']] % 10
# labels_s = (train_df[['next_week_hospitalizations']] % 100 - labels_g)//10
# labels_b = (train_df[['next_week_hospitalizations']] % 1000 - labels_s * 10 - labels_g)//100
# labels_q = (train_df[['next_week_hospitalizations']] % 10000 - labels_b * 100 - labels_s * 10 - labels_g)//1000
# labels_w = (train_df[['next_week_hospitalizations']] % 100000 - labels_q * 1000 - labels_b * 100 - labels_s * 10 - labels_g)//10000
# labels = transform_label(labels_w)
# with open("train_label_4.csv", 'w', newline='') as f:
#     fieldnames = ["0", "1", "2", "3", "4","5","6","7","8",'9']
#     writer = csv.writer(f)
#     for i in range(0,len(labels)):
#         writer.writerow(labels[i])

# ##normalize train
# train_df = read_file(train_baseline_file)
# country_df=train_df[['country']]
# categorical_columns = ['country']
# country_df = pd.get_dummies(country_df, columns=categorical_columns)
# feat = train_df.drop(['country', 'date', 'year_week'], axis=1)
# # for i in range(0,len(train_df)):
# #     data=feat.loc[i].values
# #     tmp_max=max(data)
# #     if(tmp_max>10000):
# #         data_new=data/100000
# #     elif(tmp_max>1000):
# #         data_new=data/10000
# #     elif (tmp_max > 100):
# #         data_new = data / 1000
# #     elif (tmp_max > 10):
# #         data_new = data / 100
# #     else:
# #         data_new=data/10
# #     feat.loc[i]=data_new
# feat=pd.concat([feat,country_df],axis=1)
# print(feat)
# feat.to_csv("train5.csv",index=False)
#print(labels_0[1])
# labels = train_df[['label', 'next_week_hospitalizations']]
# print(labels)
# feat = train_df.drop(['country', 'next_week_hospitalizations', 'date', 'year_week'], axis=1)

# train_df = read_file(train_baseline_file)
# x, y = train_test_split(train_df, test_size=0.1, random_state=42)
# x.to_csv('train.csv',index=False)
# y.to_csv("valiadation1.csv",index=False)
# categorical_columns = ['country']
# train_df = train_df.fillna(0)
# train_df = pd.get_dummies(train_df, columns=categorical_columns, drop_first=True)
# labels = train_df[['next_week_hospitalizations']]
# feat = train_df.drop(['next_week_hospitalizations', 'date', 'year_week'], axis=1)
#
# x_train, x_test, y_train, y_test = train_test_split(feat, labels, test_size=0.1, random_state=42)

# ##handle predict_df
# test_df = read_file(test_baseline_no_label)
# categorical_columns = ['country']
# combine_lambda = lambda x: '{} {}'.format(x.country, x.date)
# submit_df = pd.DataFrame({'country_id' : [], 'next_week_hospitalizations': []})
# submit_df['country_id'] = test_df.apply(combine_lambda, axis=1)
# #train_df["prev_daily"] = train_df.groupby(categorical_columns)["Daily hospital occupancy"].shift(7)
# test_df = test_df.fillna(0)
# test_df = pd.get_dummies(test_df, columns=categorical_columns, drop_first=True)
# feat_test = test_df.drop(['date', 'year_week'], axis=1)
# preds = clf.predict(feat_test)
# submit_df['next_week_hospitalizations'] = preds
# submit_df.to_csv('out.csv',index=False)

class CSVDataset(Dataset):
    def __init__(self, trainfile,labelfile):
        # Where the initial logic happens like reading a csv, doing data augmentation, etc.
        train_df = read_file(trainfile)
        label_df = read_file(labelfile)
        self.length=len(train_df)
        self.labels = label_df
        self.feat = train_df

    def __len__(self):
        # Returns count of samples (an integer) you have.
        return self.length

    def __getitem__(self, idx):
        # Given an index, returns the correponding datapoint.
        # This function is called from dataloader like this:
        # img, label = CSVDataset.__getitem__(99)  # For 99th item
        return self.feat.loc[idx].values,self.labels.loc[idx].values

class Feedforward(torch.nn.Module):
    def __init__(self):
        super(Feedforward, self).__init__()
        self.fc1 = torch.nn.Linear(23, 15)
        self.fc2 = torch.nn.Linear(15, 10)
        #self.fc3 = torch.nn.Linear(14, 10)
        self.relu = torch.nn.ReLU()
        self.sig=nn.Sigmoid()
    def forward(self, x):
        h1 = self.fc1(x)
        sig = self.relu(h1)
        h2 = self.fc2(sig)
        #sig = self.relu(h2)
        #output=self.fc3(sig)
        # relu3 = self.sig(h3)
        #output=self.sig(h3)
        return h2


def train(csvfile0,csvfile1,csvfile2,csvfile3,csvfile4,csvfile5,csvfile6):
    # Initialize an object of the model class
    net = Feedforward()
    # Define your loss function
    criterion = nn.MSELoss()
    # Create your optimizer
    optimizer = optim.SGD(net.parameters(),momentum=0.9,lr=0.1)
    # Initialize an object of the dataset class
    dataset = CSVDataset(csvfile0,csvfile1)
    # Wrap a dataloader around the dataset object.
    dataloader = DataLoader(dataset)
    # Beging training!

    net1 = Feedforward()
    net2 = Feedforward()
    net3 = Feedforward()
    net4= Feedforward()
    criterion1 = nn.MSELoss()
    criterion2 = nn.MSELoss()
    criterion3 = nn.MSELoss()
    criterion4 = nn.MSELoss()
    optimizer1 = optim.SGD(net.parameters(),momentum=0.9,lr=0.1)
    optimizer2 = optim.SGD(net.parameters(),momentum=0.9,lr=0.1)
    optimizer3 = optim.SGD(net.parameters(),momentum=0.8,lr=0.4)
    optimizer4 = optim.SGD(net.parameters(),lr=0.01)
    dataset1 = CSVDataset(csvfile0,csvfile2)
    dataset2 = CSVDataset(csvfile0,csvfile3)
    dataset3 = CSVDataset(csvfile0,csvfile4)
    dataset4 = CSVDataset(csvfile0,csvfile5)

    dataloader1 = DataLoader(dataset1)
    dataloader2 = DataLoader(dataset2)
    dataloader3 = DataLoader(dataset3)
    dataloader4 = DataLoader(dataset4)


    for epoch in range(10):
        # for batch_idx, (input, target) in enumerate(dataloader):
        #     # You always want to use zero_grad(), backward(), and step() in the following order.
        #     # zero_grad clears old gradients from the last step (otherwise you’d just accumulate the gradients from all loss.backward() calls).
        #     optimizer.zero_grad()
        #     # As said before, you can only code as below if your network belongs to the nn.Module class.
        #     output = net(torch.tensor(input,dtype=torch.float))
        #     #print(output)
        #     #print(output)
        #     loss = criterion(output, torch.tensor(target,dtype=(torch.float)))
        #     # loss.backward() computes the derivative of the loss w.r.t. the parameters (or anything requiring gradients) using backpropagation.
        #     loss.backward()
        #     # optimizer.step() causes the optimizer to take a step based on the gradients of the parameters.
        #     optimizer.step()
        #
        # for batch_idx, (input, target) in enumerate(dataloader1):
        #     optimizer1.zero_grad()
        #     output = net1(torch.tensor(input,dtype=torch.float))
        #     loss1 = criterion1(output, torch.tensor(target,dtype=(torch.float)))
        #     loss1.backward()
        #     optimizer1.step()
        #
        # for batch_idx, (input, target) in enumerate(dataloader2):
        #     optimizer2.zero_grad()
        #     output = net2(torch.tensor(input,dtype=torch.float))
        #     loss2 = criterion2(output, torch.tensor(target,dtype=(torch.float)))
        #     loss2.backward()
        #     optimizer.step()

        # for batch_idx, (input, target) in enumerate(dataloader3):
        #     optimizer3.zero_grad()
        #     output = net3(torch.tensor(input,dtype=torch.float))
        #     loss3 = criterion3(output, torch.tensor(target,dtype=(torch.float)))
        #     loss3.backward()
        #     optimizer3.step()

        running_loss4 = 0.0
        for i, (input, target) in enumerate(dataloader4):
            optimizer4.zero_grad()
            output = net4(torch.tensor(input,dtype=torch.float))
            loss4 = criterion4(output, torch.tensor(target,dtype=(torch.float)))
            loss4.backward()
            optimizer4.step()
            running_loss4 += loss4.item()
            if i % 500 == 499:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss4 / 500))
                running_loss4 = 0.0
        print("end of epoch {}".format(epoch+1))

    test_dp=read_file(csvfile6)
    for i in range(0,len(test_dp)):
        input=test_dp.loc[i].values
        #output_g = net(torch.tensor(input, dtype=torch.float)).data.tolist()
        #output_s = net1(torch.tensor(input, dtype=torch.float)).data.tolist()
        #output_b = net2(torch.tensor(input, dtype=torch.float)).data.tolist()
        #output_q = net3(torch.tensor(input, dtype=torch.float)).data.tolist()
        output_w = net4(torch.tensor(input, dtype=torch.float)).data.tolist()
        #pred_g=int(np.argmax(output_g))
        #pred_s=int(np.argmax(output_s))
        #pred_b=int(np.argmax(output_b))
        #pred_q=int(np.argmax(output_q))
        pred_w=int(np.argmax(output_w))
        print(pred_w)
        #print(pred_w*10000+pred_q*1000+pred_b*100+pred_s*10+pred_g)

    # output=[]
    #print(output)
    #print(mean_squared_error(output, labels))


train("train4.csv","train_label_0.csv","train_label_1.csv","train_label_2.csv","train_label_3.csv","train_label_4.csv","test.csv")
#y_pred = Feedforward("valiadation.csv")
#print(mean_squared_error(y_test, y_pred))