#!/bin/sh

python src/main.py --config=att_iql --env-config=spread with env_args.n_agents=$1 expr_name=$2_test evaluate=True test_nepisode=1 runner=episode render_gif=True checkpoint_path=results/$2 use_cuda=False runner=episode batch_size_run=1 load_step=$3 $4
