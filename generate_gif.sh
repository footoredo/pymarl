#!/bin/bash

all_args=("$@")
rest_args=("${all_args[@]:4}")

python src/main.py --config=$1 --env-config=spread with env_args.n_agents=$2 expr_name=$3_test evaluate=True test_nepisode=1 runner=episode render_gif=True checkpoint_path=results/$3 use_cuda=False runner=episode batch_size_run=1 load_step=$4 ${rest_args[@]}
