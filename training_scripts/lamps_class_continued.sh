CUDA_VISIBLE_DEVICES=5 python ../train.py \
--data_root /home/apsk14/data/final_data/Lamp/Lamp.train \
--logging_root /media/data2/sitzmann/srn-segmentation/lamps_train_continued \
--checkpoint_path /media/data2/sitzmann/srn-segmentation/lamps_train/checkpoints/epoch_0015_iter_030000.pth \
--no_validation \
--img_sidelengths 128 --max_steps_per_img_sidelength 190000 \
--batch_size_per_img_sidelength 16 \
--start_step 30000
