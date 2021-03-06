python RL/train.py --domain_name RealArm-v0 \
	--cameras 0 --frame_stack 1 --observation_type pixel --encoder_type pixel \
	--save_tb --save_buffer --save_video --save_sac \
	--work_dir real_robot_data/v0 \
	--eval_freq 150 --num_eval_episodes 1 \
	--log_interval 1 \
	--pre_transform_image_size 100 --image_size 84 --agent rad_sac --data_aug crop \
	--seed 15 \
	--batch_size 128 --num_updates 10 --num_train_steps 10000 --init_steps 0 \
	--reward_type v0 --replay_buffer_load_dir real_robot_demo/v0 \
	--synch_update --log_networks_freq 30000  --warmup_cpc 1600 --warmup_cpc_ema \
	--actor_log_std_max 0
