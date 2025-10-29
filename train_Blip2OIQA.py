import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.backends import cudnn
from scipy import stats
import numpy as np
import sys
sys.path.append('/DATA/DATA1/yangliu/code/config')
from options import *
sys.path.append('/DATA/DATA1/yangliu/code/config')
from utils import *
sys.path.append('/DATA/DATA1/yangliu/code/config')
from learning_rates import get_learning_rate_scheduler
from dataset_blip2oiqa import datasetllm, custom_collate_fn
from BLIP2OIQA_model import blip2oiqa
def loss_func(reward):
    
    # target = torch.zeros(reward.shape[0], dtype=torch.long).to(reward.device)
    # loss_list = F.cross_entropy(reward, target, reduction='none')
    
    
    reward_diff = torch.abs(reward[:, 0] - reward[:, 1])
    loss = torch.mean(reward_diff)
    #loss = reward_diff
    acc = torch.mean((reward_diff < 0.001).clone().detach().float())
    
    return loss, acc
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = datasetllm("train")
    valid_dataset = datasetllm("valid")

    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, collate_fn=custom_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=opts.batch_size, shuffle=True, collate_fn=custom_collate_fn)

    model = blip2oiqa(device).to(device)
    logfile1 = opts.logfile_name+'-moz1.txt'
    logfile2 = opts.logfile_name+'-moz2.txt'
    logfile3 = opts.logfile_name+'-moz3.txt'
    # Set the training iterations.
    opts.train_iters = opts.epochs * len(train_loader)
    steps_per_valid = len(train_loader) // opts.valid_per_epoch
    print("len(train_dataset) = ", len(train_dataset))
    print("train_dataset.iters_per_epoch = ", train_dataset.iters_per_epoch)
    print("len(train_loader) = ", len(train_loader))
    print("steps_per_valid = ", steps_per_valid)
    best_lossllm =1e9
    best_loss = 1e9
    bestrocc = 0
    bestplcc = 0
    bestkrcc = 0
    bestrmse = 0
    bestepoch = 0
    best_loss1 = 1e9
    bestrocc1 = 0
    bestplcc1 = 0
    bestkrcc1 = 0
    bestrmse1 = 0
    bestepoch1 = 0
    best_loss2 = 1e9
    bestrocc2 = 0
    bestplcc2 = 0
    bestkrcc2 = 0
    bestrmse2 = 0
    bestepoch2 = 0
    best_loss3 = 1e9
    bestrocc3 = 0
    bestplcc3 = 0
    bestkrcc3 = 0
    bestrmse3 = 0
    bestepoch3 = 0
    bestroccall = 0 
    bestplccall = 0 
    bestkrccall = 0 
    bestrmseall = 0 
    bestepochall = 0 
    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr, betas=(opts.adam_beta1, opts.adam_beta2), eps=opts.adam_eps)
    scheduler = get_learning_rate_scheduler(optimizer, opts)
    optimizer.zero_grad()
    losses = []
    acc_list = []
    losses2 = []
    acc_list2 = []
    losses3 = []
    acc_list3 = []
    lossesllm = []
    acc_listllm = []
    for epoch in range(opts.epochs):

        lossesv = []
        acc_listv = []
        y_test = []
        y_pred = []
        y_testv = []
        y_predv = []
        lossesv2 = []
        acc_listv2 = []
        y_test2 = []
        y_pred2 = []
        y_testv2 = []
        y_predv2 = []
        lossesv3 = []
        acc_listv3 = []
        y_test3 = []
        y_pred3 = []
        y_testv3 = []
        y_predv3 = []
        
        for step, batch_data_package in enumerate(train_loader):
            model.train()
           
            reward,reward2,reward3,lossllm = model(batch_data_package)
            logfile = logfile1
            best_loss = best_loss1
            bestrocc = bestrocc1
            bestplcc = bestplcc1
            bestkrcc = bestkrcc1
            bestrmse = bestrmse1
            bestepoch = bestepoch1

            yt = reward[:,0].clone().detach().float().cpu().numpy()
            yp = reward[:,1].clone().detach().float().cpu().numpy()
            yt2 = reward2[:,0].clone().detach().float().cpu().numpy()
            yp2 = reward2[:,1].clone().detach().float().cpu().numpy()
            yt3 = reward3[:,0].clone().detach().float().cpu().numpy()
            yp3 = reward3[:,1].clone().detach().float().cpu().numpy()
            y_test.extend(yt)
            y_pred.extend(yp)
            y_test2.extend(yt2)
            y_pred2.extend(yp2)
            y_test3.extend(yt3)
            y_pred3.extend(yp3)
            loss1, acc = loss_func(reward)
            loss2, acc2 = loss_func(reward2)
            loss3, acc3 = loss_func(reward3)
            # loss regularization
            loss1 = loss1 / opts.accumulation_steps
            loss2 = loss2 / opts.accumulation_steps
            loss3 = loss3 / opts.accumulation_steps
            lossllm = lossllm / opts.accumulation_steps
            # print(loss1,loss2,loss3,lossllm)
            # with open('overall7-out2.txt','a')as f:
            #     f.write(str(lossllm.clone().detach().float().cpu()))
            #     f.write('\n')
            # L = L + loss
            # back propagation
            loss =loss1+loss2+loss3
            loss.backward()
            #lossllm.backward()
            # loss2.backward()
            # loss3.backward()
            
            # if best_loss>loss:
            #     save_model_loss(model)
            #     best_loss= loss

            losses.append(loss1)
            acc_list.append(acc.item())
            losses2.append(loss2)
            acc_list2.append(acc2.item())
            losses3.append(loss3)
            acc_list3.append(acc3.item())
            lossesllm.append(lossllm)

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
                print('Iteration %d | Loss %6.5f | Acc %6.4f| Loss2 %6.5f | Acc2 %6.4f| Loss3 %6.5f | Acc3 %6.4f' % (train_iteration, sum(losses) / len(losses) , sum(acc_list) / len(acc_list), sum(losses2) / len(losses2) , sum(acc_list2) / len(acc_list2), sum(losses3) / len(losses3) , sum(acc_list3) / len(acc_list3)))
                # with open(logfile,"a") as f:
                #     f.write('Iteration %d | Loss %6.5f | Acc %6.4f| Loss2 %6.5f | Acc2 %6.4f| Loss3 %6.5f | Acc3 %6.4f' % (train_iteration, sum(losses) / len(losses) , sum(acc_list) / len(acc_list), sum(losses2) / len(losses2) , sum(acc_list2) / len(acc_list2), sum(losses3) / len(losses3) , sum(acc_list3) / len(acc_list3)))
                # with open("/DATA/DATA1/yangliu/code/output_log/test1.txt", "a") as file:
                #     # 将变量写入文件
                #     file.write(f"train_iteration : {train_iteration}\n")
                #     file.write(f"loss : {sum(losses) / len(losses)}\n")
                #     file.write(f"acc : {sum(acc_list) / len(acc_list)}\n")
                #     file.write(f"loss2 : {sum(losses2) / len(losses2)}\n")
                #     file.write(f"acc2 : {sum(acc_list2) / len(acc_list2)}\n")
                #     file.write(f"loss3 : {sum(losses3) / len(losses3)}\n")
                #     file.write(f"acc3 : {sum(acc_list3) / len(acc_list3)}\n")

                losses.clear()
                acc_list.clear()
                losses2.clear()
                acc_list2.clear()
                losses3.clear()
                acc_list3.clear()
                lossesllm.clear()
            
            # valid result print and log
            if (iterations % steps_per_valid) == 0:
                print("evalutaing",'Iteration %d')
                model.eval()
                valid_loss = []
                valid_acc_list = []
                with torch.no_grad():
                    for step, batch_data_package in enumerate(valid_loader):
                            
                        reward,reward2,reward3,lossllm = model(batch_data_package)
                        logfile = logfile1
                           
                        ytv = reward[:,0].clone().detach().float().cpu().numpy()
                        ypv = reward[:,1].clone().detach().float().cpu().numpy()
                        ytv2 = reward2[:,0].clone().detach().float().cpu().numpy()
                        ypv2 = reward2[:,1].clone().detach().float().cpu().numpy()
                        ytv3 = reward3[:,0].clone().detach().float().cpu().numpy()
                        ypv3 = reward3[:,1].clone().detach().float().cpu().numpy()
                        y_testv.extend(ytv)
                        y_predv.extend(ypv)
                        y_testv2.extend(ytv2)
                        y_predv2.extend(ypv2)
                        y_testv3.extend(ytv3)
                        y_predv3.extend(ypv3)
                          
        # model.inferenceall(batch_data_package)
        # save_model_llm(model)
        y_pred = np.array(y_pred)
        y_test = np.array(y_test)
        SROCC = stats.spearmanr(y_pred, y_test)[0]
        PLCC = stats.pearsonr(y_pred, y_test)[0]
        KROCC = stats.stats.kendalltau(y_pred, y_test)[0]
        RMSE = np.sqrt(((y_pred - y_test) ** 2).mean())
        y_predv = np.array(y_predv)
        y_testv = np.array(y_testv)
        SROCCv = stats.spearmanr(y_predv, y_testv)[0]
        PLCCv = stats.pearsonr(y_predv, y_testv)[0]
        KROCCv = stats.stats.kendalltau(y_predv, y_testv)[0]
        RMSEv = np.sqrt(((y_predv - y_testv) ** 2).mean())
        y_pred2 = np.array(y_pred2)
        y_test2 = np.array(y_test2)
        SROCC2 = stats.spearmanr(y_pred2, y_test2)[0]
        PLCC2 = stats.pearsonr(y_pred2, y_test2)[0]
        KROCC2 = stats.stats.kendalltau(y_pred2, y_test2)[0]
        RMSE2 = np.sqrt(((y_pred2 - y_test2) ** 2).mean())
        y_predv2 = np.array(y_predv2)
        y_testv2 = np.array(y_testv2)
        SROCCv2 = stats.spearmanr(y_predv2, y_testv2)[0]
        PLCCv2 = stats.pearsonr(y_predv2, y_testv2)[0]
        KROCCv2 = stats.stats.kendalltau(y_predv2, y_testv2)[0]
        RMSEv2 = np.sqrt(((y_predv2 - y_testv2) ** 2).mean())
        y_pred3 = np.array(y_pred3)
        y_test3 = np.array(y_test3)
        SROCC3 = stats.spearmanr(y_pred3, y_test3)[0]
        PLCC3 = stats.pearsonr(y_pred3, y_test3)[0]
        KROCC3 = stats.stats.kendalltau(y_pred3, y_test3)[0]
        RMSE3 = np.sqrt(((y_pred3 - y_test3) ** 2).mean())
        y_predv3 = np.array(y_predv3)
        y_testv3 = np.array(y_testv3)
        SROCCv3 = stats.spearmanr(y_predv3, y_testv3)[0]
        PLCCv3 = stats.pearsonr(y_predv3, y_testv3)[0]
        KROCCv3 = stats.stats.kendalltau(y_predv3, y_testv3)[0]
        RMSEv3 = np.sqrt(((y_predv3 - y_testv3) ** 2).mean())
        # print(len(y_pred))
        # print(len(y_predv))
        SRCCall = (SROCCv + SROCCv2 + SROCCv3)/3
        if abs(SRCCall) > abs(bestroccall):#save model for best srocc1
            save_model_srocc(model)
            bestroccall = SRCCall
            bestplccall = (PLCCv + PLCCv2 + PLCCv3)/3
            bestkrccall = (KROCCv + KROCCv + KROCCv3)/3
            bestrmseall = (RMSEv + RMSEv2 + RMSEv3)/3
            bestepochall = epoch
        # if abs(SROCCv) > abs(bestrocc):
        #     print("Best Val srocc so far. Saving model")
            bestrocc1 = SROCCv
            bestplcc1 = PLCCv
            bestkrcc1 = KROCCv
            bestrmse1 = RMSEv
            bestepoch1 = epoch
            print("best_srocc = ", bestrocc)
            # save_model_srocc(model)
        # if abs(SROCCv2) > abs(bestrocc2):
        #     print("Best Val srocc so far. Saving model")
            bestrocc2 = SROCCv2
            bestplcc2 = PLCCv2
            bestkrcc2 = KROCCv2
            bestrmse2 = RMSEv2
            bestepoch2 = epoch
            print("best_srocc2 = ", bestrocc2)
            # save_model_srocc(model)
        # if abs(SROCCv3) > abs(bestrocc3):
        #     print("Best Val srocc so far. Saving model")
            bestrocc3 = SROCCv3
            bestplcc3 = PLCCv3
            bestkrcc3 = KROCCv3
            bestrmse3 = RMSEv3
            bestepoch3 = epoch
            print("best_srocc3 = ", bestrocc3)
            # save_model_srocc(model)

        print("Epoch {} Train Results:  SROCC1={:.4f} PLCC1={:.4f} KROCC1={:.4f} RMSE1={:.4f}".format(epoch,
                                                                               SROCC,
                                                                               PLCC,
                                                                               KROCC,
                                                                               RMSE))
        
        print("Epoch {} Test Results:  SROCC={:.4f} PLCC={:.4f} KROCC={:.4f} RMSE={:.4f}".format(epoch,
                                                                               SROCCv,
                                                                               PLCCv,
                                                                               KROCCv,
                                                                               RMSEv))
        print("Epoch {} Best Results:  SROCC={:.4f} PLCC={:.4f} KROCC={:.4f} RMSE={:.4f}".format(bestepoch,
                                                                               bestrocc,
                                                                               bestplcc,
                                                                               bestkrcc,
                                                                               bestrmse))
        with open(logfile,"a") as f:
            f.write("Epoch {} Train Results1:  SROCC1={:.4f} PLCC1={:.4f} KROCC1={:.4f} RMSE1={:.4f}\n".format(epoch,
                                                                               SROCC,
                                                                               PLCC,
                                                                               KROCC,
                                                                               RMSE))
            f.write("Epoch {} Train Results2:  SROCC2={:.4f} PLCC2={:.4f} KROCC2={:.4f} RMSE2={:.4f}\n".format(epoch,
                                                                               SROCC2,
                                                                               PLCC2,
                                                                               KROCC2,
                                                                               RMSE2))
            f.write("Epoch {} Train Results3:  SROCC3={:.4f} PLCC3={:.4f} KROCC3={:.4f} RMSE3={:.4f}\n".format(epoch,
                                                                               SROCC3,
                                                                               PLCC3,
                                                                               KROCC3,
                                                                               RMSE3))
            f.write("Epoch {} Test Results1:  SROCC={:.4f} PLCC={:.4f} KROCC={:.4f} RMSE={:.4f}\n".format(epoch,
                                                                               SROCCv,
                                                                               PLCCv,
                                                                               KROCCv,
                                                                               RMSEv))
            f.write("Epoch {} Test Results2:  SROCC2={:.4f} PLCC2={:.4f} KROCC2={:.4f} RMSE2={:.4f}\n".format(epoch,
                                                                               SROCCv2,
                                                                               PLCCv2,
                                                                               KROCCv2,
                                                                               RMSEv2))
            f.write("Epoch {} Test Results3:  SROCC3={:.4f} PLCC3={:.4f} KROCC3={:.4f} RMSE3={:.4f}\n".format(epoch,
                                                                               SROCCv3,
                                                                               PLCCv3,
                                                                               KROCCv3,
                                                                               RMSEv3))
            f.write("Epoch {} Best Results:  SROCC={:.4f} PLCC={:.4f} KROCC={:.4f} RMSE={:.4f}\n".format(bestepoch,
                                                                               bestrocc1,
                                                                               bestplcc1,
                                                                               bestkrcc1,
                                                                               bestrmse1))
            f.write("Epoch {} Best Results2:  SROCC2={:.4f} PLCC2={:.4f} KROCC2={:.4f} RMSE2={:.4f}\n".format(bestepoch2,
                                                                               bestrocc2,
                                                                               bestplcc2,
                                                                               bestkrcc2,
                                                                               bestrmse2))
            f.write("Epoch {} Best Results3:  SROCC3={:.4f} PLCC3={:.4f} KROCC3={:.4f} RMSE3={:.4f}\n".format(bestepoch3,
                                                                               bestrocc3,
                                                                               bestplcc3,
                                                                               bestkrcc3,
                                                                               bestrmse3))
            f.write("Epoch {} Best Resultsall:  SROCC3={:.4f} PLCC3={:.4f} KROCC3={:.4f} RMSE3={:.4f}\n".format(bestepochall,
                                                                               bestroccall,
                                                                               bestplccall,
                                                                               bestkrccall,
                                                                               bestrmseall))
            