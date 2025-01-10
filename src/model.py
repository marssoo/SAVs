from PIL import Image
import torch
import copy
import requests
from .preprocess import *


from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import process_images, tokenizer_image_token
from qwen_vl_utils import process_vision_info


def load_image(image_file):
    try:
        image = Image.open(image_file).convert("RGB")
    except:
        return image_file
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


class llavaOVHelper:

    def __init__(self, model, tokenizer, processor, cur_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.model_config = {"n_heads":model.model.config.num_attention_heads,
                    "n_layers":model.model.config.num_hidden_layers,
                    "resid_dim":model.model.config.hidden_size,
                    "name_or_path":model.model.config._name_or_path,
                    "attn_hook_names":[f'model.layers.{layer}.self_attn.o_proj' for layer in range(model.model.config.num_hidden_layers)],
                    "layer_hook_names":[f'model.layers.{layer}' for layer in range(model.model.config.num_hidden_layers)],
                    "mlp_hook_names": [f'model.layers.{layer}.mlp.down_proj' for layer in range(model.model.config.num_hidden_layers)]}

        self.cur_dataset = cur_dataset
        self.format_func = get_format_func(cur_dataset)
        self.split_idx = 2

        self.all_heads = []
        for layer in range(28):
            for head in range(28):
                self.all_heads.append((layer, head, -1))


    def insert_image(self, text, image_list):

        conv_template = "qwen_1_5"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()


        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.model.device)

        if image_list == []:
            return (input_ids, None, None)

        image_list = load_images(image_list)
        print(image_list)
        image_sizes = [image.size for image in image_list]

        image_tensors = process_images(image_list, self.processor, self.model.config)
        image_tensors = [_image.to(dtype=torch.float16, device=self.model.device) for _image in image_tensors]

        return (input_ids, image_tensors, image_sizes)
    

    def forward(self, model_input, labels=None):

        result = self.model(model_input[0],
            images=model_input[1],
            image_sizes=model_input[2],
            labels=labels)
        return result
    

    def generate(self, model_input, max_new_tokens):

        cont = self.model.generate(
            model_input[0],
            images=model_input[1],
            image_sizes=model_input[2],
            do_sample=False,
            temperature=0,

            max_new_tokens=max_new_tokens,
        )
        
        return self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]