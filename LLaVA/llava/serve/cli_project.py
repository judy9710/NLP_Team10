import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

from PIL import Image
from transformers import TextStreamer
import matplotlib.pyplot as plt


class LLaVA(object):
    def __init__(self, model_path, device=0, model_base="lmsys/vicuna-13b-v1.5"):
        self.model_path = model_path
        self.model_base = model_base
        self.device = f"cuda:{device}"
        self.temperature = 0.2
        self.max_new_tokens = 512
        self.load_8bit, load_4bit = False, False
        self.setup()
    
    def setup(self):
        disable_torch_init()
        self.model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, context_len = load_pretrained_model(self.model_path, self.model_base, self.model_name, self.load_8bit, self.load_4bit, device=self.device)
        if 'llama-2' in self.model_name.lower():
            self.conv_mode = "llava_llama_2"
        elif "v1" in self.model_name.lower():
            self.conv_mode = "llava_v1"
        elif "mpt" in self.model_name.lower():
            self.conv_mode = "mpt"
        else:
            self.conv_mode = "llava_v0"
        return 
    
    def to_pil_image(self, matplotlib_image):
        numpy_image = matplotlib_image.to_array()
        pil_image = Image.fromarray(numpy_image)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        return pil_image
    
    def get_input(self, state):
        if "cartpole" in self.model_path:
            inp = f"A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum is placed upright on the cart and the goal is to balance the pole by applying fixed forces in the left and right direction on the cart. We can observe the position of the cart, the velocity of the cart, the angle of the pole, and the angular velocity of the pole. The range of the position of the cart is from -2.4 to 2.4. The range of the velocity of the cart is -inf to inf. The range of the angle of the pole is -0.209 to 0.209. The range of the angular velocity of the pole is -inf to inf. In this image, the position of the cart is {state['cart_pos']}, the velocity of the cart is {state['cart_vel']}, the angle of the pole is {state['pole_angle']}, and the angular velocity of the pole is {state['pole_angular_vel']}. Then, which direction should I push the cart? Answer in 0 or 1 without any description, and 0 means left, 1 means right."
        elif "mountaincar" in self.model_path:
            inp = f"A car placed stochastically at the bottom of a sinusoidal valley, with the only possible actions being the accelerations that can be applied to the car in either direction. The goal is to strategically accelerate the car to reach the goal state on top of the right hill. We can observe the position of the car along the x-axis, and the velocity of the car. The range of the position of the car is from -1.2 to 0.6. The range of the velocity of the car is -0.07 to 0.07. In this image, the position of the car is {state['car_pos']}, the velocity of the car is {state['car_vel']}. Given an action, the mountain car follows the following transition dynamics: velocity_(t+1) = velocity_t + (action - 1) * force - cos(3 * position_(t)) * gravity, position_(t+1) = position_(t) + velocity_(t+1) where force = 0.001 and gravity = 0.0025. The collisions at either end are inelastic with the velocity set to 0 upon collision with the wall. Then, which direction should I accerlerate the car in this image? Answer in 0 or 1 or 2 without any description, 0 means accelerate to the left, 1 means don't accelerate, and 2 means accelerate to the right. "    
        elif "pendulum" in self.model_path:
            inp = f"The system consists of a pendulum attached at one end to a fixed point, and the other end being free. The pendulum starts in a random position and the goal is to apply torque on the free end to swing it into an upright position, with its center of gravity right above the fixed point. We can observe the x position of the pendulum, the y position of the pendulum, and the angular velocity of the pendulum. The range of the x position of the pendulum is from -1 to 1. The range of the y position of the pendulum is from -1 to 1. The range of the angular velocity of the pendulum is -8.0 to 8.0. In this image, the x position of the pendulum is {state['x']}, the y position of the pendulum is {state['y']}, and the angular velocity of the pendulum is {state['angular_vel']}. Then, how much torque should I apply to the pendulum? Answer in a float number between -2.0 and 2.0 without any description."
        return inp

    def get_response(self, image, state):
        conv = conv_templates[self.conv_mode].copy()

        image = self.to_pil_image(image)
        image_tensor = process_images([image], self.image_processor, self.model.config)
        inp = self.get_input(state)

        if type(image_tensor) is list:
            image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
        
        inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        return outputs
