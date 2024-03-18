import pickle
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np
import pandas as pd
#import packages
import sys,os
import torch.nn as nn
#from BiGCN.Process.process import *
from torch_scatter import scatter_mean, scatter_max, scatter_add
import torch.nn.functional as F
from earlystopping import EarlyStopping
from torch_geometric.data import DataLoader
from tqdm import tqdm
#this one can be replaced by the default package from torch
from evaluate import *
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn import GCNConv,GraphConv,GINConv,GATConv
import copy
from graphmodel import *
from graphDataProcessor import *


def loadfolddata(datasetname):
  cc_path = '/graphData/'+datasetname
  train_file_path = cc_path+'_x_train.pkl'
  test_file_path = cc_path+'_x_test.pkl'
  with open(train_file_path,'rb') as f:
	trainlist = pickle.load(f)
  with open(test_file_path,'rb') as ftest:
	testlist = pickle.load(ftest)
  return trainlist,testlist



#load graphDataset to dataList
def loadNewBiData(fold_x_train, fold_x_test):
	data_path = '/graphData/'
	print("loading train set", )
	traindata_list = GraphDataset(fold_x_train, data_path=data_path)
	print("train no:", len(traindata_list))
	print("loading test set", )
	testdata_list = GraphDataset(fold_x_test, data_path=data_path)
	print("test no:", len(testdata_list))
	return traindata_list, testdata_list


def train(device,x_train,x_test,lr, weight_decay,patience,n_epochs,batchsize):
	model = SimpleGATBERTNet(768,240,64,pooling='scatter_mean').to(device)
	print(model)
	BU_params=list(map(id,model.gnn.conv1.parameters()))
	BU_params += list(map(id, model.gnn.conv2.parameters()))
	BU_params += list(map(id, model.gnn.bert.parameters())) #set BERT learning rate 
	#BU_params += list(map(id, model.gnn.conv3.parameters()))
	base_params=filter(lambda p:id(p) not in BU_params,model.parameters())
	optimizer = torch.optim.Adam([
		{'params':base_params},
		{'params':model.gnn.conv1.parameters(),'lr':lr/5},
		{'params': model.gnn.conv2.parameters(), 'lr': lr/5},
		{'params': model.gnn.bert.parameters(), 'lr': 2e-5}, #set up learning rate for BERT layers
		#{'params': model.gnn.conv3.parameters(), 'lr': lr/5}
	], lr=lr, weight_decay=weight_decay)
	model.train()
	train_losses = []
	val_losses = []
	train_accs = []
	val_accs = []
	early_stopping = EarlyStopping(patience=patience, verbose=True)
	traindata_list, testdata_list = loadNewBiData(x_train, x_test)
	train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=5)
	test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=False, num_workers=5)
	
	for epoch in range(n_epochs):  
		avg_loss = []
		avg_acc = []
		batch_idx = 0
		tqdm_train_loader = tqdm(train_loader)
		for Batch_data in tqdm_train_loader:
			Batch_data.to(device)
			dataList = Batch_data.to_data_list()
			out_labels= model(Batch_data)
			finalloss=F.nll_loss(out_labels,Batch_data.y)
			loss=finalloss
			optimizer.zero_grad()
			loss.backward()
			avg_loss.append(loss.item())
			optimizer.step()
			_, pred = out_labels.max(dim=-1)
			correct = pred.eq(Batch_data.y).sum().item()
			train_acc = correct / len(Batch_data.y)
			avg_acc.append(train_acc)
			print("Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(epoch, batch_idx,
																								 loss.item(),
																								 train_acc))
			batch_idx = batch_idx + 1

		train_losses.append(np.mean(avg_loss))
		train_accs.append(np.mean(avg_acc))

		temp_val_losses = []
		temp_val_accs = []
		temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
		temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
		temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
		temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
		model.eval()
		tqdm_test_loader = tqdm(test_loader)
		for Batch_data in tqdm_test_loader:
			optimizer.zero_grad()
			Batch_data.to(device)
			val_out = model(Batch_data)
			val_loss  = F.nll_loss(val_out, Batch_data.y)
			temp_val_losses.append(val_loss.item())
			_, val_pred = val_out.max(dim=1)
			correct = val_pred.eq(Batch_data.y).sum().item()
			val_acc = correct / len(Batch_data.y)
			Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(
				val_pred, Batch_data.y)
			temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
				Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
			temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
				Recll2), temp_val_F2.append(F2), \
			temp_val_Acc3.append(Acc3), temp_val_Prec3.append(Prec3), temp_val_Recll3.append(
				Recll3), temp_val_F3.append(F3), \
			temp_val_Acc4.append(Acc4), temp_val_Prec4.append(Prec4), temp_val_Recll4.append(
				Recll4), temp_val_F4.append(F4)
			temp_val_accs.append(val_acc)
		val_losses.append(np.mean(temp_val_losses))
		val_accs.append(np.mean(temp_val_accs))
		print("Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(epoch, np.mean(temp_val_losses),
																		   np.mean(temp_val_accs)))

		res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
			   'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
													   np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
			   'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
													   np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
			   'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
													   np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
			   'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4),
													   np.mean(temp_val_Recll4), np.mean(temp_val_F4))]
		print('results:', res)
		early_stopping(np.mean(temp_val_losses), np.mean(temp_val_accs), np.mean(temp_val_F1), np.mean(temp_val_F2),
					   np.mean(temp_val_F3), np.mean(temp_val_F4), model, 'BiGCN', 'PHEME')
		accs =np.mean(temp_val_accs)
		F1 = np.mean(temp_val_F1)
		F2 = np.mean(temp_val_F2)
		F3 = np.mean(temp_val_F3)
		F4 = np.mean(temp_val_F4)
		if early_stopping.early_stop:
			print("Early stopping")
			accs=early_stopping.accs
			F1=early_stopping.F1
			F2 = early_stopping.F2
			F3 = early_stopping.F3
			F4 = early_stopping.F4
			break
		torch.cuda.empty_cache()
	return train_losses , val_losses ,train_accs, val_accs,accs,F1,F2,F3,F4



def main():
  datasetname="pheme"
  foldnum = 0

  lr=0.0005
  weight_decay=1e-5
  patience=10
  n_epochs=200
  batchsize=2

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  test_accs = []
  NR_F1 = []
  FR_F1 = []
  TR_F1 = []
  UR_F1 = []

  train_data,test_data = loadfolddata(datasetname,foldnum)
  train_losses, val_losses, train_accs, val_accs0, accs0, F1_0, F2_0, F3_0, F4_0 = train(device,train_data,test_data,
																						 lr, weight_decay,
																						 patience,
																						 n_epochs,
																						 batchsize)


if __name__ == "__main__":
	main()

