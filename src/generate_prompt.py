import json
import os
from data.dataset import *
import argparse
from tqdm import tqdm

'''
-------------fine-tuning data format-------------
=================================================
[
    {
        'id': id
        'image': an image related to conversation
        'conversations': 
        [
            {
                'from': 'human' or 'gpt'
                'value': 'string'
            },
            ...
        ]
    },
    ...
]
'''



def generate_single_prompt(args, data):
    dataset_type = args.dataset
    question = {'from': 'human'}
    answer = {'from': 'gpt'}
    if dataset_type == 'CartPole':
        cartpole_question_template = f"A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum is placed upright on the cart and the goal is to balance the pole by applying fixed forces in the left and right direction on the cart. We can observe the position of the cart, the velocity of the cart, the angle of the pole, and the angular velocity of the pole. The range of the position of the cart is from -2.4 to 2.4. The range of the velocity of the cart is -inf to inf. The range of the angle of the pole is -0.209 to 0.209. The range of the angular velocity of the pole is -inf to inf. In this image, the position of the cart is {data['state']['cart_pos']}, the velocity of the cart is {data['state']['cart_vel']}, the angle of the pole is {data['state']['pole_angle']}, and the angular velocity of the pole is {data['state']['pole_angular_vel']}. Then, which direction should I push the cart? Answer in 0 or 1 without any description, and 0 means left, 1 means right."
        question['value'] = cartpole_question_template
        answer['value'] = str(data['action'])
    elif dataset_type == 'MountainCar':
        mountaincar_question_template = f"A car placed stochastically at the bottom of a sinusoidal valley, with the only possible actions being the accelerations that can be applied to the car in either direction. The goal is to strategically accelerate the car to reach the goal state on top of the right hill. We can observe the position of the car along the x-axis, and the velocity of the car. The range of the position of the car is from -1.2 to 0.6. The range of the velocity of the car is -0.07 to 0.07. In this image, the position of the car is {data['state']['car_pos']}, the velocity of the car is {data['state']['car_vel']}. Given an action, the mountain car follows the following transition dynamics: velocity_(t+1) = velocity_t + (action - 1) * force - cos(3 * position_(t)) * gravity, position_(t+1) = position_(t) + velocity_(t+1) where force = 0.001 and gravity = 0.0025. The collisions at either end are inelastic with the velocity set to 0 upon collision with the wall. Then, which direction should I accerlerate the car in this image? Answer in 0 or 1 or 2 without any description, 0 means accelerate to the left, 1 means don't accelerate, and 2 means accelerate to the right. "    
        question['value'] = mountaincar_question_template
        answer['value'] = str(data['action'])
    elif dataset_type == 'Pendulum':
        pendulum_question_template = f"The system consists of a pendulum attached at one end to a fixed point, and the other end being free. The pendulum starts in a random position and the goal is to apply torque on the free end to swing it into an upright position, with its center of gravity right above the fixed point. We can observe the x position of the pendulum, the y position of the pendulum, and the angular velocity of the pendulum. The range of the x position of the pendulum is from -1 to 1. The range of the y position of the pendulum is from -1 to 1. The range of the angular velocity of the pendulum is -8.0 to 8.0. In this image, the x position of the pendulum is {data['state']['x']}, the y position of the pendulum is {data['state']['y']}, and the angular velocity of the pendulum is {data['state']['angular_vel']}. Then, how much torque should I apply to the pendulum? Answer in a float number between -2.0 and 2.0 without any description."
        question['value'] = pendulum_question_template
        answer['value'] = str(data['action'])
       
    return [question, answer]
    

def generate_finetuning_data(args, dataset):
    return_list = []
    idx = 0
    for data in dataset:
        tmp_dict = {'id': str(idx), 'image': os.path.basename(data['image']), 'conversations': generate_single_prompt(args, data)}
        return_list.append(tmp_dict)
        idx += 1
    
    return return_list

def main(args):
    if args.dataset == 'CartPole':
        dataset = CartPoleDataset()
    elif args.dataset == 'MountainCar':
        dataset = MountainCarDataset()
    elif args.dataset == 'Pendulum':
        dataset = PendulumDataset()
    
    fine_tuning_list = generate_finetuning_data(args, dataset)
    print(fine_tuning_list[1])
    os.makedirs('fine_tuning', exist_ok=True)
    with open(os.path.join('fine_tuning', args.dataset + '.json'), 'w') as f:
        json.dump(fine_tuning_list, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CartPole', choices=['CartPole', 'MountainCar', 'Pendulum'])
    args = parser.parse_args()

    main(args)