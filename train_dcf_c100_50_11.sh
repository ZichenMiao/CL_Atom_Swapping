# FOR RESNET-32
#####
# Test case for ensemble training with same data for all members
#####
python train_dcfens_cifar100_CI.py \
--dataset cifar100 \
--data_path Datasets/CIFAR100 \
--num_task 11 \
--first_task_cls 50 \
--model resnet32 \
--num_member 2 \
--num_bases 12 \
--train_batch 64 \
--test_batch 128 \
--lr 0.01 \
--wd 5e-3 \
--total_epoch 250 \
--lr_schedule 100-200 \
--lr_sub 0.01 \
--wd_sub 5e-3 \
--total_epoch_sub 250 \
--lr_schedule_sub 100-200 \
--gpu $1 \
--mloss_type atoms_norm \
--mloss_weight 1.0 \
--init_with_pre \
# --label_smoothing 0.05
# --is_test \
# --start_from 1 \
# --optim nestrov
