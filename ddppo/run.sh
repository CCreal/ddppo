echo "example ant"


for ((i=0;i<5;i+=1))
do 
    python main.py --exploration_init --cuda --automatic_entropy_tuning True \
    --env-name AntTruncatedObs-v2 \
    --num_steps 150000 \
    --start_steps 5000 \
    --weight_grad 0.1 \
    --batch_size_policy 256 \
    --lr 3e-4 \
    --update_policy_times 10 \
    --updates_per_step 10 \
    --rollout_max_length 1 \
    --max_train_repeat_per_step 10 \
    --min_pool_size 5000 \
    --near_n 5 \
    --seed $i \
    --H 4
done

echo "done"