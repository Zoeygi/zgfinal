import os
import torch
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset


# 当前文件所在路径
cur_folder_path=os.path.dirname(os.path.abspath(__file__))


class VideoDataset(Dataset):
    def __init__(self,dataset_path,image_path,clip_len):
        # 数据集地址
        self.dataset_path = dataset_path
        # 训练 验证 测试集名字
        self.split=image_path
        # 生成数据的深度
        self.clip_len = clip_len

        self.resize_height,self.resize_width =171,171
        #self.resize_height,self.resize_width =128,171
        self.crop_size=112


        # 读取train val test下各个类别的行为动作
        # 每个标签下的视频已被处理为单个图片集
        # 将对应标签的视频文件名作为标签保存到labels列表中，
        # 对应标签下视频拆分后图片文件路径保存到self.fnames列表，标签与数据一一对应
        folder=os.path.join(self.dataset_path,image_path)
        self.fnames,labels=[],[]
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder,label)):
                self.fnames.append(os.path.join(folder,label,fname))
                labels.append(label)
        print(f"number of {image_path} video: {len(self.fnames)}") 

        # 获取对应数据帧标签 将标签转换为int类型 同时转换为array类型
        self.label2index={label:index for index,label in enumerate(sorted(set(labels)))}
        self.label_array=np.array([self.label2index[label] for label in labels],dtype=int)

    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, index):
        # 加载类别中的动作数据集 转为（帧 高 宽，通道）
        buffer=self.load_frames(self.fnames[index])
        # 数据深度 高宽方向随机裁剪 将加载数据转换
        buffer=self.crop(buffer,self.clip_len,self.crop_size)
        # 对模型进行归一化处理
        #buffer=self.normalize(buffer)
        # 对模型进行转换
        buffer=self.to_tensor(buffer)

        # 获取对应数据标签
        labels=np.array(self.label_array[index])

        # 返回torch格式的特征与标签
        return torch.from_numpy(buffer),torch.from_numpy(labels)

    def load_frames(self,file_dir):
        # 将文件夹下数据集排序
        frames=sorted(os.path.join(file_dir,img) for img in os.listdir(file_dir))
        # 获取该动作数据集的长度
        frame_count=len(frames)
        # 生成一个空的（frame_count,resized_height,resize_width,3)的数据
        buffer=np.empty((frame_count,self.resize_height,self.resize_width,3),np.dtype("float32"))
        # 遍历循环获得表情路径
        for i,frame_name in enumerate(frames):
            # 用cv读取图片数据 转换为np.array数据
            frame=np.array(cv.imread(frame_name)).astype(np.float64)
            #不断遍历循环赋值给buffer
            buffer[i]=frame
        return buffer
    
    def crop(self,buffer,clip_len,crop_size):
        # 在深度 高度 宽度方向生成随机长度
        time_index=np.random.randint(buffer.shape[0]-clip_len)
        height_index=np.random.randint(buffer.shape[1]-crop_size)
        width_index=np.random.randint(buffer.shape[2]-crop_size)
        # 利用切片在视频上提取，获得一个（clip_len,112,112,3)提取
        buffer=buffer[time_index:time_index+clip_len,
                      height_index:height_index+crop_size,width_index:width_index+crop_size,:]

        return buffer
    
    def normalize(self,buffer):
        # 归一化
        for i,frame in enumerate(buffer):
            frame-=np.array([[90.0,98.0,102.0]])
            buffer[i]=frame
            
        return buffer
    
    def to_tensor(self,buffer):
        # 进行维度转换
        return buffer.transpose((3,0,1,2))
    
if __name__=="__main__":
    from torch.utils.data import DataLoader
    train_data=VideoDataset(dataset_path=os.path.join(cur_folder_path,"videodata/labelled_data"),image_path="train",clip_len=16)
    train_loader=DataLoader(train_data,batch_size=32,shuffle=True,num_workers=2)

    val_data=VideoDataset(dataset_path=os.path.join(cur_folder_path,"videodata/labelled_data"),image_path="val",clip_len=16)
    val_loader=DataLoader(val_data,batch_size=32,shuffle=True,num_workers=2)

    test_data=VideoDataset(dataset_path=os.path.join(cur_folder_path,"videodata/labelled_data"),image_path="test",clip_len=16)
    test_loader=DataLoader(test_data,batch_size=32,shuffle=True,num_workers=2)
