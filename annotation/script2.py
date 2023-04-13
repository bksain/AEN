from glob import glob
import os

your_dataset_path = "E:/emotion_dataset/RAF-DB/basic/Image/aligned"
all_txt_file = glob(os.path.join('set*.txt'))
labels = ['Happy','Sad','Neutral','Angry','Surprise','Disgust','Fear'] # 0 ~ 6

labels_b = ['Surprise','Fear','Disgust','Happy','Sad','Angry','Neutral'] # 1 ~ 7

def update(file, old_str, new_str):
    file_data = ""
    with open(file, "r") as f:
        cnt = 0
        for line in f:
            label = (new_str + '/' + line)[-2]
            # if label == '1':
            #     label = 4
            # elif label == '2':
            #     label = 6
            # elif label == '3':
            #     label = 5
            # elif label == '4':
            #     label = 0
            # elif label == '5':
            #     label = 1
            # elif label == '6':
            #     label = 3
            # else:
            #     label = 2

            # new_str2 = (new_str + '/' + line)
            # new_str2 = new_str2[:-7] + '_aligned' + new_str2[-7:-2] + '1 ' + '%d'%label + new_str2[-1]
            new_str2 = (line)
            new_str2 = new_str2[:-7] + '_aligned' + new_str2[-7:-2] + label + new_str2[-1]

            line = line.replace(line,new_str2)
            file_data += line

            cnt += 1

            if cnt % 1000 == 0:
                print(cnt)
    with open('new.txt', "w") as f:
        f.write(file_data)


for txt_file in all_txt_file:
    update(txt_file, "your_dataset_path/DFEW/train/", your_dataset_path)
