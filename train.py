import torch
import csv
import random
from sklearn.model_selection import train_test_split

X = []
y = []

name = 'homo'

csvfile = open('./data/' + name + '.csv')
csvreader = csv.reader(csvfile, delimiter = '\t')

for rows in csvreader:
    X.append(torch.load('./data/esm_data/' + rows[0] + '.pt')['mean_representations'][33])
    y.append(1)

name = 'ecoli'

csvfile = open('./data/' + name + '.csv')
csvreader = csv.reader(csvfile, delimiter = '\t')

for rows in csvreader:
    X.append(torch.load('./data/esm_data/' + rows[0] + '.pt')['mean_representations'][33])
    y.append(0)

X_tensor = torch.stack((X))

print(X[0])

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y, test_size=0.2, random_state=114514)

print("Training set: ", len(X_train),"Testing set: ", len(X_test))

# from sklearn import svm

# #Create a svm Classifier
# clf = svm.SVC()

# #Train the model using the training sets
# clf.fit(X_train, y_train)

# #Predict the response for test dataset
# y_pred = clf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

import torch.nn as nn
import numpy as np

# Define the model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1280, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        x = torch.sigmoid(x)
        return x

# Define optimizer and loss functions
model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()
mean_train_losses = []
mean_valid_losses = []
valid_acc_list = []
epochs = 15

# Train the model
for epoch in range(epochs):
    model.train()
    train_losses = []
    valid_losses = []
    optimizer.zero_grad()
        
    outputs = model(X_train)
    loss = loss_fn(outputs, torch.FloatTensor(y_train).reshape(1593,1))
    loss.backward()
    optimizer.step()
        
    train_losses.append(loss.item())
            
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        outputs = model(X_test)
        loss = loss_fn(outputs, torch.FloatTensor(y_test).reshape(399,1))
            
        valid_losses.append(loss.item())
            
        accuracy = metrics.accuracy_score(y_test, torch.round(outputs))
            
    mean_train_losses.append(np.mean(train_losses))
    mean_valid_losses.append(np.mean(valid_losses))
    
    valid_acc_list.append(accuracy)
    print('epoch : {}, train loss : {:.4f}, valid loss : {:.4f}, valid acc : {:.2f}%'\
         .format(epoch+1, np.mean(train_losses), np.mean(valid_losses), accuracy))