import os
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from ivy_vl.model import IvyVLModel
from ivy_vl.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.utils import rank0_print

def load_pretrained_model_for_ivy_vl(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", torch_dtype="float16", attn_implementation="flash_attention_2", customized_config=None, overwrite_config=None, **kwargs):
    kwargs["device_map"] = device_map

    # Handle 8-bit or 4-bit quantization options
    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    elif torch_dtype == "float16":
        kwargs["torch_dtype"] = torch.float16
    elif torch_dtype == "bfloat16":
        kwargs["torch_dtype"] = torch.bfloat16
    else:
        raise ValueError("Unsupported torch_dtype")

    if customized_config is not None:
        kwargs["config"] = customized_config

    if "ivy_vl" in model_name.lower():
        # Load Ivy_VL model
        rank0_print(f"Loading Ivy_VL model from {model_path}...")

        if model_base is None:
            raise ValueError("You must provide `model_base` for Ivy_VL model.")

        # Load tokenizer for Ivy_VL model
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        
        # Load model configuration
        ivy_config = AutoConfig.from_pretrained(model_path)
        
        # Initialize Ivy_VL model
        model = IvyVLModel.from_pretrained(model_base, config=ivy_config, attn_implementation=attn_implementation, **kwargs)
        
        # Adjust token embeddings if necessary
        token_num, token_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, token_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, token_dim, device=model.device, dtype=model.dtype))

        # Load additional model weights if available
        rank0_print("Loading additional Ivy_VL weights...")
        if os.path.exists(os.path.join(model_path, "non_lora_trainables.bin")):
            non_lora_trainables = torch.load(os.path.join(model_path, "non_lora_trainables.bin"), map_location="cpu")
        else:
            from huggingface_hub import hf_hub_download
            def load_from_hf(repo_id, filename):
                cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
                return torch.load(cache_file, map_location="cpu")
            non_lora_trainables = load_from_hf(model_path, "non_lora_trainables.bin")
        
        # Modify keys to match model weights
        non_lora_trainables = {(k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)

        rank0_print("Ivy_VL model loaded successfully.")
        
    else:
        raise ValueError(f"Model {model_name} not supported for Ivy_VL")

    image_processor = None

    # Add special tokens for multimodal tasks
    if "ivy_vl" in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        # Handle vision tower if multimodal
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != "auto":
            vision_tower.to(device="cuda", dtype=torch.float16)
        image_processor = vision_tower.image_processor

    # Set context length based on model's configuration
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    elif hasattr(model.config, "max_position_embeddings"):
        context_len = model.config.max_position_embeddings
    elif hasattr(model.config, "tokenizer_model_max_length"):
        context_len = model.config.tokenizer_model_max_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
