CUDA_VISIBLE_DEVICES=9 python ../train.py --data_root /home/apsk14/data/final_data/Lamp/Lamp.train \
--logging_root /home/sitzmann/data/srn-segmentation/lamps_train_vanilla --class_weight=0. --no_validation
