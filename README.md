# AEN

The implementation of "Frame Level Emotion Guided Dynamic Facial Expression Recognition with Emotion Grouping", IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshop (CVPRW), 2023 (Accepted)

- prototype code of our method
- real implementation code (To do)

# Training

# Test
pretrained model can be download below link
https://drive.google.com/drive/folders/1_0mbYxnFY5o9oYpSTZN8s2X6J3-s9CHh?usp=sharing



# Abstract

Facial expression recognition (FER) has received considerable attention in computer vision, with ``in-the-wild" environments such as human-computer interaction and video understanding. Recognizing dynamic facial expressions in videos is generally considered a more practical and reliable approach than still images. However, the dynamic FER problem in videos has challenges in terms of both data acquisition and the structural aspects of the learning model. In particular, video frames that deviate from the target facial expression class can significantly degrade the performance of dynamic FER. In this paper, we present an affectivity extraction network (AEN) for dynamic FER. AEN combines features of different semantic levels and classifies both sentiment and specific emotion categories with emotion grouping. To address the challenges of dynamic FER, we propose frame-level emotion-guided loss functions and a structural aspect of the learning model. The AEN has two branches: a bottom-up branch that learns facial expressions representation at different semantic levels and outputs pseudo labels of facial expressions for each frame using a 2D FER model, and a top-down branch that learns discriminative representations by combining feature vectors of each semantic level for recognizing facial expressions at the corresponding emotion group. Additionally, the proposed frame-level emotion-guided loss functions encourage AEN to prevent the loss of emotional information and retain the emotional probability of a video clip. Experimental results on various video datasets show that the proposed AEN consistently outperforms the state-of-the-art in Ekman and sentiment FER. Representative results demonstrate the promise of the proposed AEN for dynamic FER in the video.

Thanks to the work of Zengqun Zhao, the code of this repository borrow from his Former-DFER repository. (https://github.com/zengqunzhao/Former-DFER)
