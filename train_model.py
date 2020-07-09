
import time 
b = time.time()
import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F 
import torch.nn as nn 
import torch.optim as optim
import matplotlib.pyplot as plt 
import cv2 
import pickle
import random 
import numpy as np
from scipy.optimize import curve_fit
from tqdm.notebook import tqdm
import numpy as np 
import os 
from pathlib import Path
a = time.time()
print('Imports complete in {} seconds.'.format(a-b))

# This following part is if you are training your model in Google Colab
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import pickle
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# Update your pickled dataset to your google drive and get shareable link id
pickle_import = drive.CreateFile({'id':'sharable_link_to_pickle_file'})

pickle_import.GetContentFile('sample0.pickle')
with open('sample0.pickle','rb') as pic:
  Data = pickle.load(pic)
random.shuffle(Data)
train_data = Data[:int(len(Data)*0.8)]
test_data = Data[int(len(Data)*0.8):]
print(f'Data points in the training data = {len(train_data)}')
print(f'Data points in the testing data = {len(test_data)}')

def plot_loss(epochs, train_loss, val_loss):
  # For plotting training and testing metrics
    plt.style.use('fivethirtyeight')
    plt.plot(epochs, train_loss, linewidth = 0.8, color = 'r', label = 'train_loss')
    plt.plot(epochs, val_loss, linewidth = 0.8, color = 'b', label = 'val_loss')
    plt.axhline(y = min(val_loss), linewidth = 0.8, color = 'g' ,label = 'min_loss')
    plt.axvline(x = epochs[val_loss.index(min(val_loss))], linewidth = 0.8, color = 'g')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation loss')
    plt.show()

def plot_acc(epochs, train_acc, val_acc, val_loss):
  # For plotting training and testing metrics
    plt.style.use('fivethirtyeight')
    plt.plot(epochs, train_acc, linewidth = 0.8, color = 'r', label = 'train_acc')
    plt.plot(epochs, val_acc, linewidth = 0.8, color = 'b', label = 'val_acc')
    plt.axhline(y = max(val_acc), linewidth = 0.8, color = 'm' ,label = 'max_acc')
    plt.axvline(x = epochs[val_loss.index(min(val_loss))], linewidth = 0.8, color = 'g', label = 'min_loss')
    plt.axvline(x = epochs[val_acc.index(max(val_acc))], linewidth = 0.8, color = 'm')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Acc')
    plt.show()

def loss_minima(loss_list): #Inputs: loss array
    _loss = sorted(loss_list)
    return _loss[0]

def one_hot(k, N_OUTPUTS):
  # For one_hot encoding a value
    a = []
    for i in range(N_OUTPUTS):
        if i != k:
            a.append(0)
        else:
            a.append(1)
    return torch.tensor(a)

def Train_optimizer(val_loss_list):
  # For early stopping to avoid over-fitting
    temp = val_loss_list
    val_loss_list = val_loss_list[-10:]
    if len(val_loss_list) <= 6:
        if min(val_loss_list) == val_loss_list[len(val_loss_list)-1]:
            if SAVE_MODEL is True:
                net.to(cpu)
                if Path(f'./{checkpoint_filename}').is_file():
                    os.remove(f'./{checkpoint_filename}')
                    if verbose:
                        print('Previous model removed.')
                    else:
                        pass
                    pickle.dump(net, open(f'./{checkpoint_filename}', 'wb'))
                    if verbose:
                        mem_size = os.stat(f'./{checkpoint_filename}').st_size
                        print(f'New model pickled. Reference epoch = {len(val_loss_list)}.')
                        print(f'Size of the model file = {mem_size//(1024**2)} MB. {(mem_size%(1024**2))/1024} KB.')
                    else:
                        pass
                    net.to(device)
                    
                else:
                    pickle.dump(net, open(f'./{checkpoint_filename}', 'wb'))
                    if verbose:
                        mem_size = os.stat(f'./{checkpoint_filename}').st_size
                        print(f'Model pickled. Reference epoch = {len(val_loss_list)}.')
                        print(f'Size of the model file = {mem_size//(1024**2)} MB. {(mem_size%(1024**2))/1024} KB.')
                    else:
                        pass
                    net.to(device)   
                    
            else:
                pass
        else:
            return False
        return True
    else:
        def new_minima_found():
            new_loss = val_loss_list[len(val_loss_list)-1]
            compare_arr = val_loss_list[:(len(val_loss_list)-1)]
            flag = False
            if new_loss < min(compare_arr):
                flag = True
                if SAVE_MODEL is True:
                    net.to(cpu)
                    if Path(f'./{checkpoint_filename}').is_file():
                        os.remove(f'./{checkpoint_filename}')
                        if verbose:
                            print('Previous model removed.')
                            print(f'New minima = {new_loss}.')
                        else:
                            pass
                        pickle.dump(net, open(f'./{checkpoint_filename}', 'wb'))
                        if verbose:
                            print(f'New model pickled. Reference epoch = {len(temp)}.')
                            mem_size = os.stat(f'./{checkpoint_filename}').st_size
                            print(f'Size of the model file = {mem_size//(1024**2)} MB. {(mem_size%(1024**2))/1024} KB.')
                        else:
                            pass
                        net.to(device)
                        
                    else:
                        pickle.dump(net, open(f'./{checkpoint_filename}', 'wb'))
                        if verbose:
                            print(f'Model pickled. Reference epoch = {len(temp)}.')
                            mem_size = os.stat(f'./{checkpoint_filename}').st_size
                            print(f'Size of the model file = {mem_size//(1024**2)} MB. {(mem_size%(1024**2))/1024} KB.')
                        else:
                            pass
                        net.to(device)
                        
                else:
                    pass
            else:
                pass
            return flag

        def change_in_slope_sign():
            def fit(x,a,b,c,d,e,f):
                return a*(x**5)+b*(x**4)+c*(x**3)+d*(x**2)+e*(x**1)+f

            def fit_slope(x,a,b,c,d,e,f):
                return 5*a*(x**4)+4*b*(x**3)+3*c*(x**2)+2*d*(x**1)+e

            if len(val_loss_list) <= 10:
                x = [i+1 for i in range(len(val_loss_list))] 
                y = val_loss_list
            else:
                x = [i+val_loss_list[0] for i in range(10)]
                y = val_loss_list
            params, foo = curve_fit(fit, x, y)
            x_fit = [i for i in np.arange(x[0], x[len(x)-1]+0.1, 0.1)]
            y_fit = [fit(i, *params) for i in np.arange(x[0], x[len(x)-1]+0.1, 0.1)]
            slopes = [fit_slope(i, *params) for i in np.arange(x[0], x[len(x)-1]+0.1, 0.1)]
            positive_slopes = []
            negative_slopes = []
            for i in range(len(slopes)):
                if slopes[i] > 0:
                    positive_slopes.append(slopes[i])
                elif slopes[i] < 0:
                    negative_slopes.append(slopes[i])
            if len(positive_slopes) == 0:
                return False
            else:
                if slopes.index(min(positive_slopes)) > slopes.index(max(negative_slopes)):
                    return True
                else:
                    return False
            
        def decreasing_loss():
            def fit(x,a,b,c,d,e,f):
                return a*(x**5)+b*(x**4)+c*(x**3)+d*(x**2)+e*(x**1)+f

            def fit_slope(x,a,b,c,d,e,f):
                return 5*a*(x**4)+4*b*(x**3)+3*c*(x**2)+2*d*(x**1)+e+f*0
            if len(val_loss_list) <= 10:
                x = [i+1 for i in range(len(val_loss_list))] 
                y = val_loss_list
            else:
                x = [i+len(temp)-9 for i in range(10)]
                y = val_loss_list
            params, foo = curve_fit(fit, x, y)
            slopes = [fit_slope(i, *params) for i in np.arange(x[0], x[len(x)-1]+0.1, 0.1)]
            flag = False
            for i in range(len(slopes)):
                if slopes[i] < 0:
                    flag = True
                    break
                else:
                    pass
            return flag
                

        if decreasing_loss() and change_in_slope_sign() and not new_minima_found():
            return False
        elif decreasing_loss() and change_in_slope_sign() and new_minima_found():
            return True
        elif decreasing_loss() and not change_in_slope_sign() and new_minima_found():
            return True
        elif not decreasing_loss():
            return False
        else:
            return False


IMG_HEIGHT, IMG_WIDTH = 192, 341
N_C = [1, 512, 256, 128, 64]
N_K = [5, 4, 3, 2]
N_MP = None
Max_mp = max(N_K)
N_HL = [512, 128]
N_OUTPUTS = 4

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
cpu = torch.device('cpu')

def flatten_shape(k):
    prod = 1
    for i in range(len(k)):
        prod = prod*k[i]
    return prod
    
def get_maxpool_params(temp):
    def fit(x,a,b,c,d):
        return a*(x**3)+b*(x**2)+c*(x**1)+d
    y = N_K
    x = [i for i in range(len(y))]
    params, foo = curve_fit(fit, x, y)
    r_squared = []
    for i in range(len(temp)):
        sum_ = 0
        for j in range(len(N_K)):
            sum_ = sum_ + (temp[i][j]-fit(j, *params))**2
        r_squared.append(sum_)
    return temp[r_squared.index(min(r_squared))]

class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        flag = False
        if N_MP is None:
            temp = []
            printed = False
            print('Configuring Hyper-parameters.')
            for a in tqdm(range(1,Max_mp+1,1)):
                for b in range(1,Max_mp+1-1,1):
                    for c in range(1,Max_mp+1-2,1):
                        for d in range(1,Max_mp+1-3,1):
                            try:
                                class temp_conv(nn.Module):
                                    def __init__(selftemp):
                                        super(temp_conv, selftemp).__init__()
                                        selftemp.conv1 = nn.Conv2d(N_C[0], N_C[1], N_K[0])
                                        selftemp.conv2 = nn.Conv2d(N_C[1], N_C[2], N_K[1])
                                        selftemp.conv3 = nn.Conv2d(N_C[2], N_C[3], N_K[2])
                                        selftemp.conv4 = nn.Conv2d(N_C[3], N_C[4], N_K[3])
                                        temp = []
                            
                                    def temp_forward(selftemp,t):
                                        bef = time.time()
                                        t = t.view(-1,1,IMG_HEIGHT,IMG_WIDTH)
                                        t = F.max_pool2d(selftemp.conv1(t),(a,a))
                                        t = F.max_pool2d(selftemp.conv2(t),(b,b))
                                        t = F.max_pool2d(selftemp.conv3(t),(c,c))
                                        t = F.max_pool2d(selftemp.conv4(t),(d,d))
                                        temp.append([a,b,c,d])
                                        aft = time.time()
                                        #print(f'Each pass takes {aft-bef} seconds.')
                                temp_net = temp_conv().to(device)
                                t = torch.rand(IMG_HEIGHT, IMG_WIDTH).to(device)
                                temp_net.temp_forward(t)
                            except:
                                pass
            self.mp = get_maxpool_params(temp)                
        else:
            self.mp = N_MP
        print(temp)
        print(f'Max_Pool params = {self.mp}')
        self.conv1 = nn.Conv2d(N_C[0], N_C[1], N_K[0])
        self.conv2 = nn.Conv2d(N_C[1], N_C[2], N_K[1])
        self.conv3 = nn.Conv2d(N_C[2], N_C[3], N_K[2])
        self.conv4 = nn.Conv2d(N_C[3], N_C[4], N_K[3])
        print('ConvNet created.')
        if flag is False:
            t = torch.rand(IMG_HEIGHT,IMG_WIDTH)
            t = t.view(-1,1,IMG_HEIGHT,IMG_WIDTH)
            t = F.max_pool2d(self.conv1(t),(self.mp[0],self.mp[0]))
            t = F.max_pool2d(self.conv2(t),(self.mp[1],self.mp[1]))
            t = F.max_pool2d(self.conv3(t),(self.mp[2],self.mp[2]))
            t = F.max_pool2d(self.conv4(t),(self.mp[3],self.mp[3]))
            flag = True
            t_shape = t.shape
        self.Flattened_input_shape = flatten_shape(t_shape)
        self.fc1 = nn.Linear(self.Flattened_input_shape, N_HL[0])
        self.fc2 = nn.Linear(N_HL[0], N_HL[1])
        self.fc3 = nn.Linear(N_HL[1], N_OUTPUTS)
        print('LinearNet added.')

    def forward(self,x):
        x = x.view(-1,1,IMG_HEIGHT,IMG_WIDTH)
        x = F.max_pool2d(self.conv1(x),(self.mp[0],self.mp[0]))
        x = F.max_pool2d(self.conv2(x),(self.mp[1],self.mp[1]))
        x = F.max_pool2d(self.conv3(x),(self.mp[2],self.mp[2]))
        x = F.max_pool2d(self.conv4(x),(self.mp[3],self.mp[3]))
        x = x.view(-1,self.Flattened_input_shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim = 1)
        return x

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
cpu = torch.device('cpu')


net = Net().to(device)
optimizer = optim.Adam(net.parameters(), lr = 0.001)
loss_fn = nn.CrossEntropyLoss()
train_data = train
test_data = test


THRESH_COUNT = 3
PLT_SHOW = 4
SAVE_MODEL = False
verbose = True
checkpoint_filename = 'fileName.pickle'

def batches(train, epoch):
    num_batches = 30
    batch_size = int(len(train)/num_batches)
    n = epoch % num_batches
    if n == 0:
        n = 1
    else:
        pass
    return train[((n-1)*batch_size):(n*batch_size)]

print(f'Running on {device}.\n')
train_flag = True
save_model_ref = 0
epoch_count = 1
epoch_list = []
val_loss_list = []
train_loss_list = []
train_acc_list = []
acc_list = []
thresh_count = THRESH_COUNT
b = time.time()
while train_flag:
    if verbose:
        disable_tqdm = False
    else:
        disable_tqdm = True 
    temp_arr = []
    temp_arr1 = []
    correct = 0
    total = 0
    train_correct = 0
    train_total = 0
    if verbose:
        print(f'\n----------Training epoch {epoch_count}----------')
    else:
        pass
    train_bef = time.time()
    train_ = batches(train_data, epoch_count)
    for data in tqdm(train_, disable = disable_tqdm):
        X, y = data
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        out = net(X)
        if torch.argmax(out) == y:
            train_correct += 1
        train_total += 1
        train_loss = loss_fn(out, y)
        train_loss.backward()
        optimizer.step()
        temp_arr1.append(train_loss.item())
    train_acc_list.append(train_correct/train_total)
    train_aft = time.time()
    temp_arr1 = torch.tensor(temp_arr1)
    l_train_data = len(train_)
    train_loss_list.append(torch.mean(temp_arr1[-int(l_train_data*0.25):]))
    if verbose:
        print(f'Training loss of epoch {epoch_count} = {torch.mean(temp_arr1[-int(l_train_data*0.25):]).item()}.')
        print(f'----------Testing epoch {epoch_count}----------')
    elif verbose is not True and epoch_count < 2:
        print(f'**Each epoch takes {round(float(train_aft - train_bef)/60.0, 2)} Minutes to train.')
    with torch.no_grad():
        test_bef = time.time()
        for data in tqdm(test_data, disable = disable_tqdm):
            X, y = data
            X = X.to(device)
            y = y.to(device)
            out = net(X)
            if(torch.argmax(out) == y):
                correct += 1
            total += 1
            val_loss = loss_fn(out, y)
            temp_arr.append(val_loss.to(cpu).item())
    test_aft = time.time()
    val_loss_list.append(np.average(temp_arr))
    epoch_list.append(epoch_count)
    acc_list.append(correct/total)
    if verbose is not True and epoch_count < 2:
        print(f'**Each epoch takes {round(float(test_aft - test_bef)/60.0, 2)} Minutes to test.')
    else:
        pass
    print(f'Validation accuracy of epoch {epoch_count} = {correct/total*100} Percent.')
    print(f'Validation loss of epoch {epoch_count} = {np.average(temp_arr)}.\n')
    if epoch_count%PLT_SHOW == 0:
        plot_loss(epoch_list, train_loss_list, val_loss_list)
    epoch_count += 1
    temp_flag = Train_optimizer(val_loss_list)
    if thresh_count == 0 and temp_flag is False:
        train_flag = False
    if train_loss == 0:
        break
    if temp_flag is False:
        if verbose:
            print('****Threshold detected****')
        else:
            pass
        thresh_count = thresh_count - 1
    else:
        thresh_count = THRESH_COUNT
a = time.time()
print('Training Terminated.')
if SAVE_MODEL is True:
    print(f'Model trained and saved. Reference epoch = {val_loss_list.index(min(val_loss_list))+1}.')
else:
    print(f'Model trained. Reference for loss minima-epoch = {val_loss_list.index(min(val_loss_list))+1}')
print(f'Total time for training = {int((a-b)/3600)} Hrs. {int((a-b)/60)} Min. {round((a-b)%60, 2)} Seconds.')
print(f'Validation accuracy = {acc_list[val_loss_list.index(min(val_loss_list))]*100} Percent.')
plot_loss(epoch_list, train_loss_list, val_loss_list)
plot_acc(epoch_list, train_acc_list, acc_list, val_loss_list)


