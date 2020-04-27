#!/bin/sh

python src/main.py --config=att_iql --env-config=spread with env_args.n_agents=3 expr_name=spread_2_iql_att_iql_on_3 evaluate=True test_nepisode=1 render_gif=True checkpoint_path=results/spread_2_att_iql use_cuda=False runner=episode batch_size_run=1 load_step=1000025
