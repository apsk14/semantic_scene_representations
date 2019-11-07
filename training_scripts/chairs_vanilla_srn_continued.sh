CUDA_VISIBLE_DEVICES=7 python ../train.py --data_root /home/apsk14/data/final_data/Chair/Chair.train \
--logging_root /media/data2/sitzmann/srn-segmentation/chairs_train_vanilla_continued \
--checkpoint_path /media/data2/sitzmann/srn-segmentation/chairs_train_vanilla/checkpoints/epoch_0005_iter_030000.pth \
--class_weight=0. --no_validation \
--img_sidelengths 128 --max_steps_per_img_sidelength 190000 --batch_size_per_img_sidelength 16 \
--start_step 30000
