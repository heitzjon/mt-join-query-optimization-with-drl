# Join Query Optimization with Deep Reinforcement Learning
This repository contains the DRL-based FOOP-environment:

"Join Query Optimization with Deep Reinforcement Learning" 
by Jonas Heitz and Kurt Stockinger

## Basics
The source code is based on the gym from [OpenAI]( https://github.com/openai/gym). The code is divided in to two parts (Agent and Environment).
<img src="https://miro.medium.com/max/1808/1*WOYVzYnF-rbdcgZU2Wt9Yw.png" alt="Agent-Environment Feedback Loop" width="500"/>

### Environment
* In the folder `/gym/envs/database/` are the reinforcement learning environments defined to plan queries according to the template of gym.
* In folder `/queryoptimization/` you find the files `QueryGraph.py` and `cm1_postgres_card.py`. The first takes over the parsing of simple SQL-Queries and includes the logic of the query planning. Whereas `cm1_postgres_card.py` delivers the expected costs of a query object according to the cost model introduced in the paper “How good are query optimizers, really?” by Leis et al.

### Agent
We used [Ray RLLib]( https://ray.readthedocs.io/en/latest/rllib.html) to train our deep reinforcement learning models. Therefore, you find in folder `agents/run/` the files:
* `config.py`: With the configurations of the models vanilla DQN (SIMPLE_CONFIG), DDQN (DOUBLE_PRIO) and PPO (PPO_CONFIG).
* `execute.py`: Includes the code to execute a set of experiments
* `models.py`: Includes the neural nets with the action-masking layer.
* `masking_env_cros.py`: Prepares the environments to deliver the information needed for the action-masking layer in `models.py`.

In the folder `/agents/rollout/` you find the scripts to test trained models and `/agents/queries/` contain the queries used for the experiments.

## Installation
1. Install PostrgreSQL
1. Load IMDB according to the guide from the [JOB](https://github.com/gregrahn/join-order-benchmark)
1. Install Python 3.*
1. Clone repository
1. Install virtual environment from `requirements.txt`
1. As a last step you need to update the DB connection details and the path of the query files in the `__init__()` and `reset()` function of the environment files at `/gym/envs/database/`.

## Run
With the script `simple_corridor.py`  in  `/agents/run/` you can check if the installation of gym and ray works.
To execute the experiments you can start `execute.py`  in `/agents/run/`.
