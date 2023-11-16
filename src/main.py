import os
import argparse

from models import OurRecommender
from trainers import PairwiseTrainer
from utils import set_seed, get_seq_dic, get_dataloder, get_rating_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default=os.environ.get("AMLT_DATA_DIR", "data"), type=str)
parser.add_argument("--output_dir", default=os.environ.get("AMLT_OUTPUT_DIR", "output_tmp"), type=str)
parser.add_argument("--data_name", default="Beauty", type=str)
parser.add_argument("--do_eval", action="store_true")
parser.add_argument("--full_sort", type=int, default=0)
parser.add_argument("--max_seq_length", default=50, type=int)

# model args
parser.add_argument("--model_name", default="CONV", type=str)
# parser.add_argument("--model_name", default="SyntV1", type=str)
parser.add_argument("--hidden_size", default=64, type=int, help="hidden size of model")
parser.add_argument("--num_hidden_layers", default=2, type=int, help="number of filter-enhanced blocks")
# parser.add_argument("--num_attention_heads", default=2, type=int)
parser.add_argument("--num_attention_heads", default=1, type=int)
parser.add_argument("--hidden_act", default="gelu", type=str) # gelu relu
parser.add_argument("--no_filter", action='store_true') # gelu relu
parser.add_argument("--attention_probs_dropout_prob", default=0.5, type=float)
parser.add_argument("--hidden_dropout_prob", default=0.5, type=float)

# train args
parser.add_argument("--lr", default=0.001, type=float, help="learning rate of adam")
parser.add_argument("--ablate", default=0, type=int, help="learning rate of adam")
parser.add_argument("--batch_size", default=256, type=int, help="number of batch_size")
parser.add_argument("--drop_null", default=0, type=int)
parser.add_argument("--zero_pad", default=1, type=int, help='if 1, then the padding input will be zero')

parser.add_argument("--patience", default=10, type=int, help="how long to wait after last time validation loss improved")
parser.add_argument("--seed", default=2, type=int)
parser.add_argument("--epochs", default=200, type=int, help="number of epochs")
parser.add_argument("--weight_decay", default=0.0, type=float, help="weight_decay of adam")

# for ablation and poolingformer
parser.add_argument("--n_fft", default=50, type=int)
parser.add_argument("--local", default=3, type=int)
parser.add_argument("--pool_size", default=2, type=int)
parser.add_argument("--use_causalmask", default=0, type=int)
parser.add_argument("--switch_order", default=0, type=int)

# for conv
parser.add_argument("--conv_name", default=1, type=int)
parser.add_argument("--initialize", default=1, type=int)
parser.add_argument("--padding_mode", default=1, type=int)
parser.add_argument("--conv_size", default=30, type=int)
parser.add_argument("--act", default=0, type=int) # drop
parser.add_argument("--ffn_multiplier", default=4, type=int, help="try to decrease the parameters in FFN")

parser.add_argument("--trial", default='None')

args = parser.parse_args()
print(args)

set_seed(args.seed)

seq_dic, max_item = get_seq_dic(args)
args.item_size = max_item + 1


train_dataloader, eval_dataloader, test_dataloader = get_dataloder(args,seq_dic)

model = OurRecommender(args=args)
trainer = PairwiseTrainer(model, train_dataloader, eval_dataloader, test_dataloader, args)

if args.full_sort:
    args.valid_rating_matrix, args.test_rating_matrix = get_rating_matrix(args.data_name, seq_dic, max_item)

if args.do_eval:
    checkpoint_path = ''
    trainer.load(checkpoint_path)
    print(f"Load model from {checkpoint_path} for test!")
    scores, result_info = trainer.test(0, full_sort=args.full_sort)

else:
    trainer.train()
