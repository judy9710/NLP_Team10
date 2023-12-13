# Can The Intelligence of LLM Solve the Real World Problems?
> NLP Team 10  
> Amy Li, Hyoyoung Lho, Jinmo Kim, Joonghyuk Shin, Junseok Lee
---
## Install
1. Clone this repository, and navigate to the repository directory
```Bash
git clone git@github.com:judy9710/NLP_Team10.git
cd NLP_Team10
```
2. Make conda environment & install package for LLaVA
```Bash
conda create -n nlp_team10 python=3.10 -y
conda activate nlp_team10
pip install --upgrade pip
cd LLaVA && pip install -e .
```
3. Install additional packages for training cases (LLaVA)
```Bash
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```
4. Install packages for Gymnasium
```Bash
cd .. && pip install -r requirements.txt
```

## How to use
0. Create dataset with RL algorithm (PPO/DQN/DDPG)
```Bash
cd utils
python training_models.py --env CartPole # or MountainCar, Pendulum
```
1. Create prompt for fine-tuning LLaVA
```Bash
cd ../src
python generate_prompt.py --dataset CartPole # or MountainCar, Pendulum
```
2. Fine-tune LLaVA with LoRA
```Bash
cd ../LLaVA
bash ./scripts/v1_5/finetune_lora_cartpole.sh
# bash ./scripts/v1_5/finetune_lora_mountaincar.sh
# bash ./scripts/v1_5/finetune_lora_pendulum.sh
```
3. Evaluate
```Bash
python llava/serve/cli_project.py --env cartpole # or mountaincar, pendulum
```