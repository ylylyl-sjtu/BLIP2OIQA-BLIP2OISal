import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn
import torch.nn as nn
from scipy import stats
import numpy as np
import sys
sys.path.append('/DATA/DATA1/yangliu/code/config')
from options import *
sys.path.append('/DATA/DATA1/yangliu/code/config')
from utils import *
sys.path.append('/DATA/DATA1/yangliu/code/config')
from learning_rates import get_learning_rate_scheduler
from dataset_blip2oisal import datasetllm, custom_collate_fn
from BLIP2OISal_model import blip2oisal
import os

EPSILON = 1e-8
def normalize(x, method='standard', axis=None):
	x = np.array(x, copy=False)
	if axis is not None:
		y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
		shape = np.ones(len(x.shape))
		shape[axis] = x.shape[axis]
		if method == 'standard':
			res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
		elif method == 'range':
			res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
		elif method == 'sum':
			res = x / np.float_(np.sum(y, axis=1).reshape(shape))
		else:
			raise ValueError('method not in {"standard", "range", "sum"}')
	else:
		if method == 'standard':
			res = (x - np.mean(x)) / np.std(x)
		elif method == 'range':
			res = (x - np.min(x)) / (np.max(x) - np.min(x))
		elif method == 'sum':
			res = x / float(np.sum(x))
		else:
			raise ValueError('method not in {"standard", "range", "sum"}')
	return res

def CC(pred, gt):
	pred = pred.clone().cpu()
	gt = gt.clone().cpu()
	map1 = np.array(pred.detach(), copy=False)
	map2 = np.array(gt.detach(), copy=False)
	# Normalize the two maps to have zero mean and unit std
	map1 = normalize(map1, method='standard')
	map2 = normalize(map2, method='standard')
	# Compute correlation coefficient
	return np.corrcoef(map1.ravel(), map2.ravel())[0,1]

def MSE(pred, gt):
	mse_loss = nn.MSELoss()
	return mse_loss(pred, gt)

def KLLoss(map_pred, map_gtd):
    map_pred = map_pred.float()
    map_gtd = map_gtd.float()
    
    map_pred = map_pred.view(1, -1) # change the map_pred into a tensor with n rows and 1 cols
    map_gtd = map_gtd.view(1, -1) # change the map_pred into a tensor with n rows and 1 cols

    min1 = torch.min(map_pred)
    max1 = torch.max(map_pred)
    # print("min1 and max1 are :", min1, max1)
    map_pred = (map_pred - min1) / (max1 - min1 + EPSILON) # min-max normalization for keeping KL loss non-NAN

    min2 = torch.min(map_gtd)
    max2 = torch.max(map_gtd)
    # print("min2 and max2 are :", min2, max2)
    map_gtd = (map_gtd - min2) / (max2 - min2 + EPSILON) # min-max normalization for keeping KL loss non-NAN

    map_pred = map_pred / (torch.sum(map_pred) + EPSILON)# normalization step to make sure that the map_pred sum to 1
    map_gtd = map_gtd / (torch.sum(map_gtd) + EPSILON) # normalization step to make sure that the map_gtd sum to 1
    # print("map_pred is :", map_pred)
    # print("map_gtd is :", map_gtd)


    KL = torch.log(map_gtd / (map_pred + EPSILON) + EPSILON)
    # print("KL 1 is :", KL)
    KL = map_gtd * KL
    # print("KL 2 is :", KL)
    KL = torch.sum(KL)
    # print("KL 3 is :", KL)
    # print("KL loss is :", KL)

    return KL

def loss_func(pred, gt):
	CC_loss = CC(pred, gt)
	MSE_loss = MSE(pred, gt)
	KLD_loss = KLLoss(pred, gt)
	total_loss = 0.2*(1-CC_loss) + 0*MSE_loss + 0.2*KLD_loss
	return total_loss, CC_loss, MSE_loss, KLD_loss


if __name__ == "__main__":
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

    train_dataset = datasetllm("train")
    valid_dataset = datasetllm("valid")

    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=opts.batch_size, shuffle=True, collate_fn=custom_collate_fn)

    model = mintiqa(device).to(device)
    log_folder = '/DATA/DATA1/yangliu/code/saliency_log_folder'
    logfile1 = os.path.join(log_folder, opts.logfile_name+'-loss.txt')
    # Set the training iterations.
    opts.train_iters = opts.epochs * len(train_loader)
    steps_per_valid = len(train_loader) // opts.valid_per_epoch
    print("len(train_dataset) = ", len(train_dataset))
    print("train_dataset.iters_per_epoch = ", train_dataset.iters_per_epoch)
    print("len(train_loader) = ", len(train_loader))
    print("steps_per_valid = ", steps_per_valid)
    best_loss = 1e9
    best_cc = 1e9
    best_mse = 1e9
    best_kld = 1e9
    bestepoch = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, betas=(opts.adam_beta1, opts.adam_beta2), eps=opts.adam_eps)
    scheduler = get_learning_rate_scheduler(optimizer, opts)
    optimizer.zero_grad()
    # losses = []
    for epoch in range(opts.epochs):
        losses = []
        lossesv = []
        cc_losses = []
        cc_lossesv =[]
        mse_losses = []
        mse_lossesv = []
        KLD_losses = []
        KLD_lossesv =[]
        losses_all = []
        cc_losses_all = []
        mse_losses_all = []
        KLD_losses_all = []
        
        for step, batch_data_package in enumerate(train_loader):
            model.train()
           
            pred, gt = model(batch_data_package)
            loss, cc_loss, mse_loss, kld_loss = loss_func(pred, gt)
            logfile = logfile1
            best_loss = best_loss
            bestepoch = bestepoch

            loss = loss / opts.accumulation_steps
           
            loss.backward()
            #clear every accumulation_steps
            losses.append(loss)
            cc_losses.append(cc_loss)
            mse_losses.append(mse_loss)
            KLD_losses.append(kld_loss)
			#save for one epoch
            losses_all.append(loss)
            cc_losses_all.append(cc_loss)
            mse_losses_all.append(mse_loss)
            KLD_losses_all.append(kld_loss)

            iterations = epoch * len(train_loader) + step + 1
            train_iteration = iterations / opts.accumulation_steps
            
            # update parameters of net 
            if (iterations % opts.accumulation_steps) == 0:
                # optimizer the net
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
                # train result print and log 
                # if get_rank() == 0:
                print('Iteration %d | Loss %6.5f| CC %6.5f | MSE %6.5f | KLD %6.5f' % (train_iteration, sum(losses) / len(losses), sum(cc_losses) / len(cc_losses), sum(mse_losses) / len(mse_losses), sum(KLD_losses) / len(KLD_losses)))
                # print(len(losses))
                # print(len(cc_losses))
                # print(len(mse_losses))#supposed to be 1
                losses.clear()
                cc_losses.clear()
                mse_losses.clear()
            
            # valid result print and log
            if (iterations % steps_per_valid) == 0:
                print("evalutaing",'Iteration %d')
                model.eval()
                with torch.no_grad():
                    for step, batch_data_package in enumerate(valid_loader):
                            
                        pred, gt = model(batch_data_package)
                        lossv, cc_lossv, mse_lossv, kld_lossv = loss_func(pred, gt)
                        logfile = logfile1
                         
                        lossesv.append(lossv)
                        cc_lossesv.append(cc_lossv)
                        mse_lossesv.append(mse_lossv)
                        KLD_lossesv.append(kld_lossv)
                print('length of lossesv', len(lossesv))
                          
        # model.inferenceall(batch_data_package)
        # save_model_llm(model)
        lossv = sum(lossesv)/len(lossesv)
        if abs(lossv) < abs(best_loss):#save model for best srocc1
            save_model_srocc(model)
            best_loss = lossv
            best_cc = sum(cc_lossesv)/len(cc_lossesv)
            best_mse = sum(mse_lossesv)/len(mse_lossesv)
            best_kld = sum(KLD_lossesv)/len(KLD_lossesv)
            bestepoch = epoch
        
            print("best_loss = ", lossv)
        
        print('length of losses all', len(losses_all))
        print("Epoch {} Train Results:  LOSS={:.4f}  CC={:.4f}  MSE={:.4f} KLD=={:.4f}".format(epoch, sum(losses_all) / len(losses_all), sum(cc_losses_all) / len(cc_losses_all), sum(mse_losses_all) / len(mse_losses_all), sum(KLD_losses_all) / len(KLD_losses_all)))
        print("Epoch {} Test Results:  LOSS={:.4f}  CC={:.4f}  MSE={:.4f}  KLD=={:.4f}".format(epoch, sum(lossesv) / len(lossesv), sum(cc_lossesv) / len(cc_lossesv), sum(mse_lossesv) / len(mse_lossesv), sum(KLD_lossesv) / len(KLD_lossesv)))
        print("Epoch {} Best Results:  LOSS={:.4f}  CC={:.4f}  MSE={:.4f}  KLD=={:.4f}".format(bestepoch, best_loss, best_cc, best_mse, best_kld))
        with open(logfile,"a") as f:
            f.write("Epoch {} Train Results:  LOSS={:.4f}  CC={:.4f}  MSE={:.4f}  KLD=={:.4f}\n".format(epoch, sum(losses_all) / len(losses_all), sum(cc_losses_all) / len(cc_losses_all), sum(mse_losses_all) / len(mse_losses_all), sum(KLD_losses_all) / len(KLD_losses_all)))
            f.write("Epoch {} Test Results:  LOSS={:.4f}  CC={:.4f}  MSE={:.4f}  KLD=={:.4f}\n".format(epoch, sum(lossesv) / len(lossesv), sum(cc_lossesv) / len(cc_lossesv), sum(mse_lossesv) / len(mse_lossesv), sum(KLD_lossesv) / len(KLD_lossesv)))
            f.write("Epoch {} Best Results:  LOSS={:.4f}  CC={:.4f}  MSE={:.4f}  KLD=={:.4f}\n".format(bestepoch, best_loss, best_cc, best_mse, best_kld))
