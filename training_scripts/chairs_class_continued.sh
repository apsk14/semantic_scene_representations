CUDA_VISIBLE_DEVICES=8 python ../train.py --data_root /home/apsk14/data/final_data/Chair/Chair.train \
--logging_root /media/data2/sitzmann/srn-segmentation/chairs_train_continued --no_validation \
--checkpoint_path /media/data2/sitzmann/srn-segmentation/chairs_train/checkpoints/epoch_0005_iter_030000.pth \
--img_sidelengths 128 --max_steps_per_img_sidelength 190000 --batch_size_per_img_sidelength 16 \
--start_step 30000
