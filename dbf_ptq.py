import argparse
import os
import json
import yaml
import time


def parse_args():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--target_bits", type=float, default=2.0)
    parser.add_argument("--model_name", type=str, default="llama2_7b")
    parser.add_argument("--model_path", type=str, default="/mnt/disk2-part1/pretrained-models-pytorch/llm/llama2_7b")
    parser.add_argument("--data_path", type=str, default="/mnt/disk2-part1/datasets/llm")
    parser.add_argument("--train_dataset", type=str, default="redpajamas")
    parser.add_argument("--seqlen", type=int, default=4096)

    # ptq params
    parser.add_argument("--only_full_ft", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--one_cycle_lr", action="store_true")
    parser.add_argument("--n_calib_data", type=int, default=256)
    parser.add_argument("--n_epochs", type=int, default=8)
    parser.add_argument("--norm_order", type=float, default=2.0)
    parser.add_argument("--n_grad_accumu", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--train_scaling_vectors", action="store_true")
    parser.add_argument("--lr_s", type=float, default=1e-4)
    parser.add_argument("--epoch_s", type=int, default=2)


    # alter optimization and admm params
    parser.add_argument("--admm_reg", type=float, default=0.03)
    parser.add_argument("--alter_opt_outer_iters", type=int, default=260)
    parser.add_argument("--alter_opt_inner_iters", type=int, default=3)

    # logging and saving 
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--is_save_ckpt", action="store_true")

    return parser.parse_args()

args = parse_args()

################ save args as yaml config file
print(vars(args))
os.makedirs(args.save_dir, exist_ok=True)
with open(os.path.join(args.save_dir, "args.yaml"), "w") as f:
    yaml.dump(vars(args), f)


################# loading model
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["OMP_NUM_THREADS"] = "1"
import torch
import torch.nn as nn
from transformers import AutoTokenizer
import sys
#sys.path.append("/mnt/disk2-part1/mingchuan/motlibs")
#from apps.llm.finetuning.utils.data import HuggingFaceDataloader
from dbf_utils import get_opt, collect_norms, opt_sequential, eval_model, BitLinear

model = get_opt(args.model_name)
tokenizer_path = args.model_name
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

print("total params: ", sum(p.numel() for p in model.parameters()), 
      "\nlinear layers params: ", sum(p.numel() for p in model.model.layers.parameters()))
print(model)

#################### loading dataset

#dataloader = HuggingFaceDataloader(
#    name = args.train_dataset,
#    split = "train", 
#    block_size = args.seqlen,
#    shuffle = True,
#    batch_size = 1,
#    use_cache = True,
#)

#dataloader.configure(datapath=args.data_path , model_name=args.model_name, tokenizer=tokenizer)
#dataloader = dataloader.get_dataloader()

#eval_dataloader = HuggingFaceDataloader(
#    name = "wikitext",
#    split = "test", 
#    block_size = args.seqlen,
#    shuffle = False,
#    batch_size = 1,
#    use_cache = True,
#)
#eval_dataloader.configure(datapath=args.data_path , model_name=args.model_name, tokenizer=tokenizer)
#eval_dataloader = eval_dataloader.get_dataloader()



n_samples = 256
import datasets
import numpy as np
ds = datasets.load_from_disk("/projects/p487-24-1/redpajama_tokenized_llama2/")

np.random.seed(47)
inds = np.random.randint(0, len(ds), size=n_samples)

dataloader = torch.LongTensor(ds[inds]["input_ids"])
dataloader.shape

import datautils
_, eval_dataloader = datautils.get_loaders(
    "wikitext2", seed=0, model=args.model_name, seqlen=model.seqlen
)
print("eval_loader", eval_dataloader.input_ids.shape)


##################### collecting norms 

# np.random.seed(47)
model = collect_norms(model, dataloader, n_calib_data=args.n_calib_data)

# check 
print("sanity check for i_norm and o_norm")
for n, m in model.named_modules():
    if type(m) == nn.Linear and "lm_head" not in n:
        print("i_norm: ", m.i_norm.shape, m.i_norm)
        print("o_norm: ", m.o_norm.shape, m.o_norm)
        break

##########################  PTQ process
if True:
    start = time.time()
    opt_sequential(model, dataloader, dev="cuda", 
                    target_bits=args.target_bits, 
                    n_samples=args.n_calib_data, 
                    norm_order=args.norm_order,
                    n_epochs=args.n_epochs, 
                    lr=args.lr, 
                    weight_decay=args.weight_decay, 
                    n_grad_accumu=args.n_grad_accumu,
                    one_cycle_lr=args.one_cycle_lr,
                    admm_reg=args.admm_reg,
                    outer_iters=args.alter_opt_outer_iters,
                    inner_iters=args.alter_opt_inner_iters,
                    save_dir=args.save_dir,
                    only_full_ft=args.only_full_ft,
                    train_scaling_vectors=args.train_scaling_vectors,
                    lr_s=args.lr_s,
                    epoch_s=args.epoch_s,
                    test_dataloader=eval_dataloader if args.eval else None,
                   )
    print("total time", time.time() - start)

if args.is_save_ckpt:
    print("saving model...")
    # convert to compact format
    for n, m in model.named_modules():
        if type(m) == BitLinear and "lm_head" not in n:
            m.pack_to_save()
    torch.save(model.state_dict(), os.path.join(args.save_dir, f"model_saved_as_state_dict.pth"))
    print(f"Model's save dict has been saved!")


########################## PPL testing and dump results 
print("Testing PPL...")
torch.cuda.empty_cache()
model.cuda()
model.gradient_checkpointing_disable()
ppl = eval_model(model, eval_dataloader)
print("Got PPL", ppl)
# os.makedirs(args.save_dir, exist_ok=True)
with open(os.path.join(args.save_dir, "test_results.json"), "w") as f:
    json.dump({"eval_perplexity": ppl, "seq_len": model.seqlen}, f)



