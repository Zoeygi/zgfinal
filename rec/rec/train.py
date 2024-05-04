import torch 
from torch import nn,optim
import C3D_model
import  S3D_model
from tensorboardX import SummaryWriter
import os
from datetime import datetime
import socket
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import VideoDataset
import torchvision
from attention import ChannelAttention,SpatialAttention

# learn_rate=1e-5
# num_epoches=10
classes=2

# 当前文件所在路径
cur_folder_path=os.path.dirname(os.path.abspath(__file__))
# 分类后的数据路径 
labelled_data_path=os.path.join(cur_folder_path,"videodata/labelled_data")

# save_dir用于保存日志
def train_model(model,device,save_dir,train_dataloader,val_dataloader,test_dataloader):

    # 定义损失 优化 学习率更新策略
    criterion=nn.CrossEntropyLoss()
    # criterion=nn.BCEWithLogitsLoss()
    # optimizer=optim.Adam(model.parameters(),lr=learn_rate)
    optimizer=optim.SGD(model.parameters(),lr=learn_rate,momentum=0.9,weight_decay=5e-4)
    schedule=optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)

    model.to(device)
    criterion.to(device)

    # 日志记录
    log_dir=os.path.join(save_dir,'model',datetime.now().strftime("%b%d_%H-%H-%S")+'_'+socket.gethostname())
    writer=SummaryWriter(log_dir=log_dir)

    # 模型训练
    trainval_loaders={"train":train_dataloader,"val":val_dataloader}
    trainval_size={x:len(trainval_loaders[x].dataset) for x in ["train", "val"]}
    test_size=len(test_dataloader.dataset)


    for epoch in range(num_epoches):
        for phase in ["train", "val"]:
            start_time = time.time()
            # 初始化损失与准确率
            running_loss=0.0
            running_corrects=0.0

            if phase=="train":
                model.train()
            else:
                model.eval()

            for inputs,labels in tqdm(trainval_loaders[phase]):
                # 数据与标签放入模型
                inputs=inputs.to(device)
                labels=labels.to(device)
                optimizer.zero_grad()


                if phase =="train":
                    outputs=model(inputs)
                else:
                    with torch.no_grad():
                        outputs=model(inputs)
                
                # 计算softmax输出概率
                probs=nn.Softmax(dim=1)(outputs)
                # probs=torch.sigmoid(outputs)
                # 计算最大概率标签
                preds=torch.max(probs,1)[1]
                labels=labels.long()

                loss=criterion(probs,labels)


                if phase=="train":
                    loss.backward()
                    optimizer.step()

                # 计算累计损失值
                running_loss+=loss.item()*inputs.size(0)

                running_corrects+=torch.sum(preds==labels.data)
            # schedule.step()
            epoch_loss=running_loss/trainval_size[phase]
            epoch_acc=running_corrects.double()/trainval_size[phase]

            if phase=="train":
                writer.add_scalar("data/train_loss_epoch",epoch_loss,epoch)
                writer.add_scalar("data/train_acc_epoch",epoch_acc,epoch)
            else:
                writer.add_scalar("data/val_loss_epoch",epoch_loss,epoch)
                writer.add_scalar("data/val_acc_epoch",epoch_acc,epoch)
            
            # 计算时间
            stop_time=time.time()

            print(f"[{phase}] Epoch:{epoch+1}/{num_epoches} loss:{epoch_loss} acc:{epoch_acc}")
            print("Execution time: " + str((stop_time - start_time) // 60) + " min " + "%.2f" % ((stop_time - start_time) % 60) + " sec\n")
    writer.close()

    torch.save({"epoch":epoch+1,"state_dict":model.state_dict(),"opt_dict":optimizer.state_dict(),},os.path.join(save_dir,"model","C3D"+"_epoch"+str(epoch)+".pth.tar"))
    print(f"save model at {os.path.join(save_dir,"model","C3D"+"_epoch"+str(epoch)+".pth.tar")}\n")

    # 模型测试
    model.eval()
    running_corrects=0.0
    print("testing:")
    # 循环推理测试集数据
    for inputs,labels in tqdm(test_dataloader):
        # 数据与标签放入模型
        inputs=inputs.to(device)
        labels=labels.long()
        labels=labels.to(device)    
        with torch.no_grad():
            outputs=model(inputs)

        # 计算softmax输出概率
        probs=nn.Softmax(dim=1)(outputs)
        # 计算最大概率标签
        preds=torch.max(probs,1)[1]
        running_corrects+=torch.sum(preds==labels.data)    
        print(f"预测值 {preds}")
        print(f"实际值 {labels.data}")
    epoch_acc=running_corrects.double()/test_size

    print(f"test acc:{epoch_acc}")
    return [epoch_acc,preds,labels.data]

if __name__ == "__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_loader=DataLoader(VideoDataset(dataset_path=labelled_data_path,image_path="train",clip_len=16),batch_size=32,shuffle=True,num_workers=0)

    
    val_loader=DataLoader(VideoDataset(dataset_path=labelled_data_path,image_path="val",clip_len=16),batch_size=36,shuffle=True,num_workers=0)

    test_loader=DataLoader(VideoDataset(dataset_path=labelled_data_path,image_path="test",clip_len=16),batch_size=36,shuffle=True,num_workers=0)


    models = {
    # 'C3D': torchvision.models.video.r3d_18(pretrained=False),
    'C3D': C3D_model.C3D(classes),
    # 'S3D':S3D_model.S3D(2),
    # 'Swin3D': torchvision.models.video.swin3d_t(pretrained=False),
    'MC3D': torchvision.models.video.mc3_18(pretrained=False)
    # Add other models as needed
    }
    accuracys={}

    for model_name,model in models.items():
        if model_name=='C3D' :
            learn_rate=1e-5
            num_epoches=15
            model.block1=nn.Sequential(
                model.block1,
                ChannelAttention(ratio=8,channel=64),
                SpatialAttention(kernel_size=7)
                )
            model.block4=nn.Sequential(
                model.block4,
                ChannelAttention(ratio=8,channel=512),
                SpatialAttention(kernel_size=7)
                )
            # model.layer2=nn.Sequential(
            #     model.layer2,
            #     ChannelAttention(ratio=8,channel=128),
            #     SpatialAttention(kernel_size=7)
            #     )
            # model.fc=nn.Linear(model.fc.in_features,2) 
        elif model_name=='MC3D':
                learn_rate=1e-4
                num_epoches=10
                model.layer1=nn.Sequential(
                model.layer1,
                ChannelAttention(ratio=8,channel=64),
                SpatialAttention(kernel_size=7)
                )
                model.layer4=nn.Sequential(
                model.layer4,
                ChannelAttention(ratio=8,channel=512),
                SpatialAttention(kernel_size=7)
                )
                model.fc=nn.Linear(model.fc.in_features,2)
        elif model_name=='Swin3D':
            model.head=nn.Linear(model.head.in_features,2)

        print("-"*50)
        print(f"{model_name} is training!\n\n")
        accuracys[model_name]=train_model(model,device=device,save_dir=os.path.join(cur_folder_path,"logs"),train_dataloader=train_loader,val_dataloader=val_loader,test_dataloader=test_loader)
        print(f"{model_name} finshed train!\n\n")

    for model_name,acc in accuracys.items():
        print(f"{model_name} test acc: {acc[0]}")
        print(f"    预测值 {acc[1]}")
        print(f"    实际值 {acc[2]}")
