#!/usr/bin/env bash
set -eux
R_DIR=`dirname $0`; MYDIR=`cd $R_DIR;pwd`
cd ${MYDIR}/../../../
# config env
source ${MYDIR}/model_conf

source ./env.sh
source ./utils.sh

check_iplist

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
output_dir=./output/${task}
log_dir=${output_dir}/log
save_model_base_dir=$output_dir/save_model
mkdir -p $output_dir $log_dir $save_model_base_dir

python3.7 -u ./src/run_myunimo.py --use_cuda "True" \
            --is_distributed ${is_distributed:-"False"} \
            --weight_sharing ${weight_sharing:-"True"} \
            --use_fast_executor ${e_executor:-"true"} \
            --use_fp16 ${use_fp16:-"false"} \
            --nccl_comm_num ${nccl_comm_num:-1} \
            --use_hierarchical_allreduce ${use_hierarchical_allreduce:-"False"} \
            --in_tokens ${in_tokens:-"false"} \
            --use_dynamic_loss_scaling ${use_fp16} \
            --init_loss_scaling ${loss_scaling:-12800} \
            --beta1 ${beta1:-0.9} \
            --beta2 ${beta2:-0.98} \
            --epsilon ${epsilon:-1e-06} \
            --verbose true \
            --do_train ${do_train:-"True"} \
            --do_val ${do_val:-"True"} \
            --do_test ${do_test:-"True"} \
            --do_pred ${do_pred:-"True"} \
            --pred_save ${pred_save:-"./output/predict/test"} \
            --batch_size ${bs:-16} \
            --init_pretraining_params ${init_model:-""} \
            --train_set ./data/SST-2/train.tsv \
            --dev_set ./data/SST-2/dev.tsv \
            --test_set ./data/SST-2/test.tsv \
            --checkpoints ${save_model_dir:-""} \
            --save_checkpoints ${save_checkpoints:-"True"} \
            --save_steps ${save_steps:-1000} \
            --weight_decay ${weight_decay:-"0.1"} \
            --warmup_proportion ${warmup_ratio:-"0.06"} \
            --validation_steps ${validation_steps:-"100"} \
            --epoch ${epoch:-1} \
            --max_seq_len ${max_len:-512} \
            --learning_rate ${lr:-"5e-5"} \
            --lr_scheduler ${lr_scheduler:-"linear_warmup_decay"} \
            --skip_steps ${skip_steps:-"10"} \
            --num_iteration_per_drop_scope 10 \
            --num_labels ${num_labels:-2} \
            --unimo_vocab_file ${vocab_file} \
            --encoder_json_file ${bpe_json} \
            --vocab_bpe_file ${bpe_file} \
            --unimo_config_path ${config_path} \
            --eval_mertrics ${eval_mertrics:-"simple_accuracy"} \
            --random_seed ${seed:-1} >> $log_dir/${log_prefix:-"myrun"}lanch.log 2>&1

