#!/bin/sh
curl --skip-existing -L -o rwf2000.zip\
  https://www.kaggle.com/api/v1/datasets/download/vulamnguyen/rwf2000
unzip -n rwf2000.zip -d dataset

curl --skip-existing -L -o ucf101-action-recognition.zip\
  https://www.kaggle.com/api/v1/datasets/download/matthewjansen/ucf101-action-recognition
unzip -n ucf101-action-recognition.zip -d dataset/UCF101

curl --skip-existing -L -o affectnet.zip\
  https://www.kaggle.com/api/v1/datasets/download/mstjebashazida/affectnet
unzip -n affectnet.zip -d dataset
cd dataset
mv 'archive (3)/Train' AffectNet/Train
mv 'archive (3)/Test' AffectNet/Test
rm -r 'archive (3)'
