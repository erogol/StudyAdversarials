'''
    Models decrease accuracy on adversarials instances as we train. It supports
    the low probability region argumnet. As the models learn the space better,
    they diverge from the low probability regions and adversarial accuracy
    goes down.
    
    Max single model adversarial performance is ~23%
    
    Mean ensemble of the model does not make any difference with ~21% accuracy.
    
    THIS IS INTRUGUING. Max ensemble of the models increases adversarial 
    performance to ~44%.
'''

# -*- coding: utf-8 -*-
from utils import load_data, plot_mnist
from nets.mnist_net import MnistNet, train, test
import torch as th
from torch import optim
from torch.utils import data
import copy
from sklearn.cross_validation import ShuffleSplit

# set default torch data type
th.set_default_tensor_type("torch.FloatTensor")

##############################################################################
# TRAIN MODELS
##############################################################################
    
ss = ShuffleSplit(60000,n_iter=20, test_size=0.2)
models = []
accus_test = []
accus_advs = []
for train_index, val_index in ss:
    
    # load pre-saved data
    x_train, y_train, x_test, y_test, x_advs = load_data()
    
    # convert to torch tensors
    x_train = th.FloatTensor(x_train[train_index])
    y_train = th.LongTensor(y_train[train_index].argmax(axis=1))
    x_test  = th.FloatTensor(x_test)
    y_test  = th.LongTensor(y_test.argmax(axis=1))
    x_advs = th.FloatTensor(x_advs)
    
    # create dataset object
    data_train = data.TensorDataset(x_train, y_train)
    data_test  = data.TensorDataset(x_test, y_test)
    data_advs  = data.TensorDataset(x_advs, y_test)
    
    # BUG
    data_train.target_tensor = data_train.target_tensor.squeeze()
    data_test.target_tensor = data_test.target_tensor.squeeze()
    data_advs.target_tensor = data_advs.target_tensor.squeeze()
    
    # create data-loader
    train_loader = data.DataLoader(data_train, batch_size=128, shuffle=True)
    test_loader = data.DataLoader(data_test, batch_size=128, shuffle=False)
    advs_loader = data.DataLoader(data_advs, batch_size=128, shuffle=False)
    
    # create network
    model = MnistNet()
    model.cuda()
    
    # create optimizer
    optimizer = optim.Adadelta(model.parameters(), lr=0.1, rho=0.95, eps=1e-8, )
    
    # train and test model
    for epoch in range(1, 15):
        train(epoch, model, optimizer, train_loader)
        print("TEST SET --")
        accu_test = test(epoch, model, test_loader)
        print("ADVS SET --")
        accu_advs = test(epoch, model, advs_loader) 
        print("\n")       
        
    models.append(copy.deepcopy(model))
    accus_test.append(accu_test)
    accus_advs.append(accu_advs)
    
print("test mean accuracy of 5 models: %f"%(th.mean(th.FloatTensor(accus_test))))
print("advs mean accuracy of 5 models: %f"%(th.mean(th.FloatTensor(accus_advs))))
    
##############################################################################
# ENSEMBLE MODELS
##############################################################################
 # load pre-saved data
x_train, y_train, x_test, y_test, x_advs = load_data()

# convert to torch tensors
y_test  = th.LongTensor(y_test.argmax(axis=1))
x_advs = th.FloatTensor(x_advs)

# create dataset object
data_advs  = data.TensorDataset(x_advs, y_test)

# BUG
data_advs.target_tensor = data_advs.target_tensor.squeeze()

# create data-loader
advs_loader = data.DataLoader(data_advs, batch_size=128, shuffle=False)    

correct = 0
ens_probs = []
for model in models:
    model.eval()
    all_probs = None
    for data, target in advs_loader:
        data, target = data.cuda(), target.cuda()
        data, target = th.autograd.Variable(data, volatile=True), th.autograd.Variable(target)
        output = model(data)
        probs = output.data
        if all_probs is None:
            all_probs = probs
        else:
            all_probs = th.cat([all_probs,probs])    
    ens_probs.append(all_probs)
    
# mean ensemble
correct = 0
mean_probs = th.zeros([10000, 10])
for probs in ens_probs:
    mean_probs = mean_probs + probs.cpu()
pred = mean_probs.max(1)[1]
correct += pred.eq(y_test).cpu().sum()
accuracy = 100. * correct / len(test_loader.dataset)
print("Mean ensemble accuracy: %f"%(accuracy))

# max ensemble (Stupid!!)
preds = th.zeros([1000])*-1000
scores = th.zeros([1000])*-1000
for c, probs in enumerate(ens_probs):
    if c == 0:
        scores, preds = probs.max(1)
    else:
        scores_model, preds_model = probs.max(1)
        scores[scores<scores_model] = scores_model[scores<scores_model]
        preds[scores<scores_model] = preds_model[scores<scores_model]

correct += preds.cpu().eq(y_test).cpu().sum()
accuracy = 100. * correct / len(test_loader.dataset)
print("Max ensemble accuracy: %f"%(accuracy))
