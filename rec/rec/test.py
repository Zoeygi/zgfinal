import torch
import cv2 as cv
import C3D_model
import numpy as np
import os

# 当前文件所在路径
cur_folder_path=os.path.dirname(os.path.abspath(__file__))


def center_crop(frame):
    frame=frame[8:120,30:142]
    return np.array(frame).astype(np.uint8)


def inference():
    device=torch.device("cuda" if torch.cuda.is_available else 'cpu')

    with open(os.path.join(cur_folder_path,"videodata","labels.txt"),"r") as f:
        class_names=f.readlines()
        f.close()

    model=C3D_model.C3D(num_classes=2)
    checkpoint=torch.load("pth.tar")
    model.load_state_dict(checkpoint['state_dict'])

    model.to(device)
    model.eval()

    video='.mp4'
    cap=cv.VideoCapture(video)
    retaining=True

    clip=[]
    while retaining:
        retaing,frame=cv.read()
        if not retaining and frame is None:
            continue
        tmp=center_crop(cv.resize(frame,(171,128)))
        tmp-=np.array([90.0,98.0,102.0])
        clip.append(tmp)

        if len(clip) == 16:
            inputs=np.array(clip).astype(np.float32)
            inputs=np.expand_dims(inputs,axis=0)

            # print(inputs.shape)
            inputs=np.transpose(0,4,1,2,3)
            inputs=torch.from_numpy(inputs).to(device)
            
            with torch.no_grad():
                outputs=model.forward(inputs)

                probs=torch.nn.Softmax(dim=1)(outputs)
                label=torch.argmax(probs,1)[1].detach().cpu().numpy()[0]
                # print(label)

                cv.putText(frame,class_names[label].split(' ')[-1].strip(),(20,20),cv.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),1)
                cv.putText(frame,"prob:%.4f"%probs[0][label],(20,40),cv.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),1)
            clip.pop(0)

        cv.imshow("result",frame)
        cv.waitKey(30)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    inference()