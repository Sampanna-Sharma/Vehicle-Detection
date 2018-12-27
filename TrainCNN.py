import torch
import torch.nn as nn
from yolo import yolo
import numpy as np

from getdata import gettrainingdata

dtype = torch.float32
device = "cpu"

X_train, X_test, Y_train, Y_test = gettrainingdata(load = True)
X_train = torch.from_numpy(X_train).view(-1,1,128,128)
Y_train = torch.tensor(torch.from_numpy(Y_train),dtype = torch.long)
X_test = torch.from_numpy(X_test).view(-1,1,128,128)
Y_test = torch.tensor(torch.from_numpy(Y_test),dtype = torch.long)

l_r = 1e-4
batch_size = 100
input_dim = 128*128
num_epoch = 10 
n_class = 3

model = yolo(3)
#model = torch.load("0model.ckpt")
criterion = nn.CrossEntropyLoss()
#criterion = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters())
#optimizer = torch.optim.SGD(model.parameters(), lr=l_r)


for epoch in range(num_epoch):
    model.train()
    for i in range(0,X_train.shape[0]-batch_size,batch_size):
        X = X_train[i:i+batch_size,:,:,:]
        Y = Y_train[i:i+batch_size,:]
        ypred = model(X).view(batch_size,n_class)
        loss = criterion(ypred, torch.max(Y,1)[1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print ('Epoch [{:.4f} of {} ], Loss: {:.4f}'.format(i/X_train.shape[0], epoch+1, loss.item()))
    
    model.eval()
    correct = 0
    for i in range(X_test.shape[0]-1):
        l = (model.forward(X_test[i:i+1,:,:,:]).data.cpu().numpy()).reshape(1,3)
        label = Y_test[i:i+1,:]
        l = np.exp(l)/np.sum(np.exp(l),1)
        if((np.argmax(label.data.cpu().numpy(),1) == np.argmax(l,1))[0]):
            correct = correct + 1
    
    print("Test Result",correct/X_test.shape[0])

    torch.save(model, str(epoch)+'model.tar')




    





        



