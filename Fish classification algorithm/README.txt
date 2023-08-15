To use InceptionV3 to train the fish classification model and integrate it into the android app, please follow the steps below. 
Note: Make sure that you have installed the latest version of Keras! 

Step1. Download the WildFish Dataset from https://uowmailedu-my.sharepoint.com/:f:/g/personal/tac247_uowmail_edu_au/EpafFIT1BCtMsCtUNepponkByb0nuV82KU0uEyFbSc9bYQ?e=PReD9F

Step2. Download the NSWFish Dataset from https://drive.google.com/drive/folders/1RvblHBRcmhTRRL3ASzAc5x9uKReccBkn?usp=sharing

Note: The downloaded WildFish and NSWFish datasets have been split into three sets including 60% train, 20% validation and 20% test sets respectively by using split_train_validation_test_wildfish.py and split_train_validation_test_nswfish.py.

Step3. Use InceptionV3_FinalModel.py to train the Inception_V3 network. Firstly, remember to change all the dataset directories to the correct ones where the downloaded datasets are stored. The best model and its weights will be saved as an h5 document. Also, the test accuracy will show after running the code. Then the saved h5 document will be converted to the "InceptionV3_weights02.tflite" file. 

Step4. The labels in the NSWFish Dataset are stored in the "labels.txt" file. This file and the tflite file can be integrated into the Android application. 


