## FOR ImageNet-50/11
python train_dcfens_imgnet100_CI.py --num_task 11 \
--first_task_cls 50 \
--list_used 1993 \
--random_classes \
--num_member 2 \
--init_with_pre \
--train_batch 128 \
--test_batch 128 \
--lr 0.1 \
--wd 5e-4 \
--total_epoch 150 \
--lr_schedule 60-100 \
--lr_sub 0.05 \
--wd_sub 5e-4 \
--total_epoch_sub 150 \
--lr_schedule_sub 70-120 \
--num_bases 12 \
--gpu $1 \
--start_from 1