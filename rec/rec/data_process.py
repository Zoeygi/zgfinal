import os
import cv2 as cv
from sklearn.model_selection import train_test_split
import mediapipe as mp
import shutil

'''
需创建数据集文件夹 videodata 在videodata下创建ori_data
videodata
'''


# 当前文件所在路径
cur_folder_path=os.path.dirname(os.path.abspath(__file__))


def process_video(ori_data_path,video,label,save_dir):
    # resized_height =128
    resized_height =171
    resized_width=171

    # 读取视频名 创建以此为名的文件夹 保存视频帧
    video_filename=video.split('.')[0]
    if not os.path.exists(os.path.join(save_dir,video_filename)):
        os.makedirs(os.path.join(save_dir,video_filename))
    
    # 读取视频，转换为numpy数组
    capture=cv.VideoCapture(os.path.join(ori_data_path,label,video))
    # 读取视频文件的帧数
    frame_count=int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    # 读取视频每帧的宽与高
    frame_width=int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height=int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))

    # 调整视频帧读取频率 保证划分视频至少有16帧
    # 每隔多少帧读取一帧
    EXTRACT_FREQUENCY=4
    while(frame_count//EXTRACT_FREQUENCY<=16):
        EXTRACT_FREQUENCY-=1


    # 目前帧索引
    count=0
    # 被保存帧数量
    i=0
    # 读取视频帧的布尔值
    retaining=True


    # 初始化mediapipe的人脸检测模块
    mp_face_detection = mp.solutions.face_detection
    # drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)


    # 运行人脸检测
    while(count<frame_count and retaining):
        # 结果值与视频帧
        retaining,frame=capture.read()
        if frame is None:
            continue
        # 根据频率读取视频帧
        if count%EXTRACT_FREQUENCY==0:
            
            # 将图像转换为RGB格式（mediapipe需要的格式）
            rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
                # 检测到的人脸结果
                results = face_detection.process(rgb_image)

                # 绘制矩形框
                if results.detections:
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        # 图片相对大小
                        ih, iw, _ = frame.shape
                        bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                        frame=frame[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
                        # cv.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)

                        if frame.size and ((frame_width!=resized_height) or (frame_height!=resized_width)):
                            frame=cv.resize(frame,(171,171))


                    # 调整视频帧大小
                    # if (frame_width!=resized_height) or (frame_height!=resized_width):
                    #     frame=cv.resize(frame,(resized_width,resized_height))

                    # # cv.imwrite(os.path.join(save_dir,video_filename,f"0000{i}.jpg"),image,[int(cv.IMWRITE_PNG_COMPRESSION),0])
                        if frame.size :
                            cv.imwrite(os.path.join(save_dir,video_filename,f"0000{i}.jpg"),frame)
                            i+=1
                    if i==50:
                        break
        count+=1
    if i<16:
        print(f"dir--{save_dir}\n label--{label}----video {video_filename}  detected faces are not enough")
        # 带有人脸图片不够 删除该文件夹
        shutil.rmtree(os.path.join(save_dir,video_filename))
    else:
        print(f"label--{label}----video {video_filename} finished split!")
    capture.release()

    
def preprocess(ori_data_path,output_data_path):
    # 若不存在训练后的数据文件夹 则创建一个
    if not os.path.exists(output_data_path):
        os.makedirs(os.path.join(output_data_path,"train"))
        os.makedirs(os.path.join(output_data_path,"val"))
        os.makedirs(os.path.join(output_data_path,"test"))

    # 遍历原始文件夹下所有类别文件
    for file in os.listdir(ori_data_path):
        file_path=os.path.join(ori_data_path,file)

        # 获取每个类别文件下所有视频名
        video_files=[name for name in os.listdir(file_path)]   
        # 划分类型下所有视频的元素 
        train_and_val,test=train_test_split(video_files,test_size=0.2)
        train,val=train_test_split(train_and_val,test_size=0.2)
        # train_and_val,test=train_test_split(video_files,test_size=0.2,random_state=42)
        # train,val=train_test_split(train_and_val,test_size=0.2,random_state=42)

        # 生成对应类别的视频文件路径
        train_dir=os.path.join(output_data_path,"train",file)
        val_dir=os.path.join(output_data_path,"val",file)
        test_dir=os.path.join(output_data_path,"test",file)
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
        if not os.path.exists(val_dir):
            os.mkdir(val_dir)
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)

            
        # 对train val test数据进行处理
        for video in train:
            process_video(ori_data_path,video,file,train_dir)
        for video in val:
            process_video(ori_data_path,video,file,val_dir)
        for video in test:
            process_video(ori_data_path,video,file,test_dir)
        print(f"{file}划分完成")
    print("所有类别划分完成")


def label_text_write(ori_data_path,out_label_path):
    folder=ori_data_path
    # 按序记录所有文件路径与标签
    fnames,labels=[],[]
    # 读取原始目录下数据标签
    for label in sorted(os.listdir(folder)):
        # 读取每个标签目录下的文件
        for fname in os.listdir(os.path.join(folder,label)):
            fnames.append(os.path.join(folder,label,fname))
            labels.append(label)
        
    # 将标签转换为对应的索引
    label2index={label:index for index,label in enumerate(sorted(set(labels)))}
    if not os.path.exists(out_label_path+"/labels.txt"):
        with open(out_label_path+"/labels.txt","w") as f:
            for id,label in enumerate(sorted(label2index)):
                f.writelines(str(id+1)+' '+label+'\n')


if __name__=="__main__":
    # 原始视频数据 需按标签分类
    ori_data_path=os.path.join(cur_folder_path,"videodata/ori_data")
    # 原始视频数据所在文件夹 用于保存标签文件
    out_label_path=os.path.join(cur_folder_path,"videodata")
    # 分类后的数据路径 
    output_data_path=os.path.join(cur_folder_path,"videodata/labelled_data")

    # # 生成标签及其对应的索引
    # label_text_write(ori_data_path,out_label_path)

    #划分数据集 生成对应图片数据集
    preprocess(ori_data_path,output_data_path)
                                    
            
            

