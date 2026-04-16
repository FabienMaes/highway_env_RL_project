# highway_env_RL_project

To train the SB3, DQN, and DDQN models on the DCE, we provide `dce_training.py`, which contains the agent definitions and the training loop. This is accompanied by `dce_training.sh`, a bash script used to execute the training process on the DCE cluster.

For our reward shaping experiments, specifically those exploring curriculum learning, we provide `aggressive_config.py` to configure the environment parameters, alongside the `test_aggressive_driver.ipynb` notebook, which contains the code to train the agent under these specific conditions.

Finally, the `All plot figure` directory contains all the generated training curves (loss, mean reward per episode, and survival time). It also includes the model bias analysis, the distribution of taken actions, and the final performance comparisons of the models evaluated across three different seeds.