from .model import *
from .preprocess import *
from tqdm import tqdm
import torch
import argparse
torch.set_grad_enabled(False)
from transformers.utils import logging
logging.set_verbosity_error() 

#parser
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="llava_ov")
parser.add_argument("--num_head", type=int, default=20)
parser.add_argument("--data_name", type=str, default="Mhalu")
parser.add_argument("--train_path", type=str, default=None)
parser.add_argument("--val_path", type=str, default=None)
parser.add_argument("--utils", type=str, default='base',
                    help='Which utils (thus head selection procedure) to chose')
parser.add_argument("--eval_zeroshot", action="store_true", 
                   help="Whether to run zero-shot evaluation")
#new arg
parser.add_argument('--quantize', type=int, default=None,
                    help="number of bits to operate quantization on.")
args = parser.parse_args()

#pick selection procedure
if args.utils == 'base':
    from .utils import *
elif args.utils == 'polar':
    from .utils_polar import *
elif args.utils == 'artanh':
    from .utils_artanh import *
else:
    raise ValueError("""Utils argument should be one of 'base', 'polar' or 'artanh'.""")

def eval_dataset(args):

    model = load_model(args.model_name, args.data_name, quantize=args.quantize)

    train_data = open_data(args.data_name, args.train_path)
    test_data = open_data(args.data_name, args.val_path)

    # Zero-shot evaluation
    if args.eval_zeroshot:
        zs_correct = 0
        for item in tqdm(test_data, desc="Zero-shot Evaluation"):
            # Create model input
            model_input = model.insert_image(item['question'], [item['image']])
            
            # Generate response and clean it
            output = model.generate(model_input, max_new_tokens=32).strip()
            
            # Direct comparison with dataset label
            if output == item['label']:
                zs_correct += 1
                
        print("Zero-shot Accuracy:", zs_correct / len(test_data))
    else:
        # SAVs embeddings
        multimodal_embeddings = mllm_encode(model, train_data, num_head=args.num_head)

        correct_count = 0
        ###Checking how well it can classify a given query
        for item in tqdm(test_data):
            cur_class = mllm_classify(item, model, multimodal_embeddings)
            if item['label'] == cur_class:
                correct_count += 1
        print("SAVs Accuracy:", correct_count / len(test_data))
        with open('results/results.csv', 'a') as f:
            f.write(','.join([
                args.model_name,
                args.data_name,
                str(args.num_head),
                args.utils,
                str(correct_count / len(test_data))
            ])+ ',,,' + '\n')


# Now run
eval_dataset(args)
