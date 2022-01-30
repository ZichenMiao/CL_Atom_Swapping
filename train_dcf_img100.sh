## FOR ImageNet-50/6
python train_dcfens_imgnet100_CI.py --num_task $1 \
--first_task_cls 50 \
--list_used 1993 \
--random_classes \
--num_member 2 \
--num_bases 12 \
--init_with_pre \
--train_batch 128 \
--test_batch 128 \
--lr 0.1 \
--wd 1e-4 \
--total_epoch 1 \
--lr_schedule 60-100 \
--lr_sub 0.05 \
--wd_sub 1e-4 \
--total_epoch_sub 1 \
--lr_schedule_sub 60-90 \
--gpu $2 \