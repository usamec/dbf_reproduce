import torch
import torch.nn as nn

# from bf16_fused_adam import BF16FusedAdamW
from tqdm import tqdm, trange
import math

from transformers import AutoModelForCausalLM, AutoConfig

import json
import os

from bf16_fused_adam import BF16FusedAdamW
from torch.utils.tensorboard import SummaryWriter

def get_opt_bad(model):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    config = AutoConfig.from_pretrained(model)
    kwargs = {
            "torch_dtype": torch.bfloat16, # fp32 or bf16, otherwise will be nans
            "low_cpu_mem_usage": True,
            "device_map": "balanced",
        }
    
    model = AutoModelForCausalLM.from_pretrained(model, config=config, **kwargs)
    model.seqlen = model.config.max_position_embeddings
    print("seqlen: ", model.seqlen)
    return model

def get_opt(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16, attn_implementation = "flash_attention_2")
    print("ms", model.config.max_position_embeddings)
    model.seqlen = model.config.max_position_embeddings
    return model



def power_iteration(A, num_iters=5):
    """
    Performs power iteration to compute the top-1 singular vectors and value.
    
    Arguments:
        A (torch.Tensor): The input matrix of shape (m, n).
        num_iters (int): Number of iterations to perform.
    
    Returns:
        u (torch.Tensor): Dominant left singular vector (m,).
        sigma (torch.Tensor): Dominant singular value (scalar).
        v (torch.Tensor): Dominant right singular vector (n,).
    """
    # Start with a random vector on the appropriate device
    n = A.shape[1]
    v = torch.randn(n, device=A.device)
    v = v / torch.norm(v)
    
    for _ in range(num_iters):
        # Multiply A*v
        u = torch.mv(A, v)
        u_norm = torch.norm(u)
        if u_norm == 0:
            break
        u = u / u_norm
        
        # Multiply A^T*u
        v = torch.mv(A.t(), u)
        v_norm = torch.norm(v)
        if v_norm == 0:
            break
        v = v / v_norm
    
    # Estimate the dominant singular value as ||A*v||
    sigma = torch.norm(torch.mv(A, v))

    # The left singular vector corresponding to sigma:
    u = torch.mv(A, v) / sigma
    return u, sigma, v

def svid_decomp(W, all_parts=False):

    """
    SVID decomposition: 
    W   = W_sign * |W| 
        ~= rank_1_approximation(W_sign) * |W| 
        = u * Sign(w) * v*T.

    Arguments:
        W - original matrix
    Returns:
        W^{hat} - approximated matrix
        or (u*s, W_sign, v) if all_parts is True
    """
    Sg = W.sign()
    Sg[Sg == 0] = 1
    u, s, v = power_iteration(W.abs(), num_iters=5)
    apx = s * torch.ger(u, v)
    
    if all_parts:
        return u * s, Sg, v
    else: 
        return apx * Sg

def admm(A, W, Z, U, reg, iters, rho_start=0.03):
    """
    ADMM procedure to solve the subproblem of alternating minimization:: min f(x), s.t. x\in C.
    Assume we fix A and optimize B, in this case would be:
    min_B ||AB-W||, s.t. B = a * B_s * m1^T

    Arguments:
        A - the matrix that we want to fix (B in the paper)
        W - scaled W matrix: W' = o * W * i^T, see 3.3 of paper
        Z - the current B matrix (from last admm iteration)
        U - the current U for B matrix (from last admm iteration)
        reg - regularization coefficient for hessian matrix
        iters - number of admm iterations

    Returns:
        Z - the optimized B matrix that minimize the subproblem 
        U - the optimized scaled dual variables U for B matrix 
    
    --------------------------------------------------------
    Note: Alternating minimization algorithm 

    init A, B with random
    for 1...outer steps:
        fix A
        for 1...inner steps:
            min_B ||AB-W||, s.t. B = b * B_s * m2^T
        fix B
        for 1...inner steps:
            min_A ||AB-W||, s.t. A = a * A_s * m1^T
        
    ------------------------------------------------------
    """
    
    XX = A.T.matmul(A) # B^T @ B
    XX += torch.diag(torch.ones_like(XX.diag())) * XX.diag().mean() * reg
    
    Wnn = W # * norm2.unsqueeze(1)
    rho = 1
    XY = A.T.matmul(Wnn) # B^T @ W
    XXinv = torch.inverse(XX + torch.eye(XX.shape[1], device=XX.device) * rho)
    XXinv2 = torch.inverse(XX + torch.eye(XX.shape[1], device=XX.device) * rho_start)
    
    B = XXinv2.matmul(XY + rho_start*(Z-U))

    # B - binary matrix W^{hat}, U - dual variables U, Z - A
    for _ in range(iters-1):
        Z = svid_decomp(B + U, all_parts=False)
        U = U + (B - Z)    
        B = XXinv.matmul(XY + rho*(Z-U))

    Z = svid_decomp(B + U, all_parts=False)
    U = U + (B - Z)
    return (Z), U

def factorizeT(W, XX, o_norm, outer_iters, inner_iters, admm_reg, target_bits):
    
    norm = XX.sqrt().unsqueeze(1) + 1e-8
    norm_o = o_norm.sqrt() + 1e-8
    Wn = W * norm * norm_o
       
    mid = int(target_bits*(W.shape[0]*W.shape[1]) / (W.shape[0] + W.shape[1]))
    
    # random initialization for A and B matrices
    Az = torch.randn((W.shape[0], mid), device=W.device)
    Au = torch.zeros_like(Az)

    Bz = torch.randn((mid, W.shape[1]), device=W.device)
    Bu = torch.zeros_like(Bz)
    
    # alternating minimization: first initialize A with a random matrix, then fix A and optimize B, then fix B and optimize A
    # we prefer to use fewer inner updates (ADMM steps) and more outer updates (alternating minimization steps)
    for itt in trange(outer_iters):
        rho_start = min(1.0, itt / (outer_iters-3))**3 # from DSF paper

        mid = Bz.norm(dim=1) + 1e-12
        Az, Au = (x.T for x in admm(Bz.T / mid, 
                                    Wn.T,
                                    Az.T, 
                                    Au.T, 
                                    reg=admm_reg,
                                    iters=inner_iters,
                                    rho_start=rho_start))
        
        mid = Az.norm(dim=0) + 1e-12
        Bz, Bu = admm(Az / mid, 
                      Wn, 
                      Bz, 
                      Bu, 
                      reg=admm_reg,
                      iters=inner_iters,
                      rho_start=rho_start)
        if itt == outer_iters - 1:
            print("err", itt, ((Az / mid).matmul(Bz) - Wn).square().sum().item(), (Wn).square().sum().item())

    # print(f"relative err percent of iter {itt}: ", ((Az / mid).matmul(Bz) - Wn).square().sum().item() / (Wn).square().sum().item())
            
    return ((Az / norm).matmul(Bz / norm_o)).T, (Bz / norm_o).T, (Az / norm).T, 1 / mid


def factorizef(W, XX, o_norm, outer_iters, inner_iters, admm_reg, target_bits):
    """
    Arguments: 
        W - input matrix.
        XX - input activation norm - as column (input) importance. 
        o_norm - gradient norm - as row (output) importance. 

    Returns:
        AB - the matmul of A and B
        A matrix = a * A_sign * m1^T
        B matrix = m2 * B_sign * b^T
        m - scaling vector m

    """
    if W.shape[0] >= W.shape[1]:
        return factorizeT(W.T, 
                          XX, 
                          o_norm, 
                          outer_iters=outer_iters, 
                          inner_iters=inner_iters,
                          admm_reg=admm_reg,
                          target_bits=target_bits)

    # see 3.3 We first calculate W' = o * W * i^T
    norm = XX.sqrt() + 1e-8
    norm_o = (o_norm.sqrt() + 1e-8).unsqueeze(1)
    Wn = W * norm * norm_o 

    # the size of middle dimension
    mid = int(target_bits*(W.shape[0]*W.shape[1]) / (W.shape[0] + W.shape[1])) 
    
    # random initialization
    Az = torch.randn((W.shape[0], mid), device=W.device)
    Au = torch.zeros_like(Az)
    Bz = torch.randn((mid, W.shape[1]), device=W.device)
    Bu = torch.zeros_like(Bz)

    ## 3.3 factorize: W' ~= (a' * A_sign * m^T)(B_sign * b'^T)
    # alternating minimization: first initialize A, B with a random matrix, then fix B and optimize A, then fix A and optimize B
    # we prefer to use fewer inner updates (ADMM steps) and more outer updates (alternating minimization steps)
    for itt in trange(outer_iters):
        rho_start = min(1.0, itt / (outer_iters-3))**3

        mid = Bz.norm(dim=1) + 1e-12 # normalize rows as described in 3.2
        Az, Au = (x.T for x in admm(Bz.T / mid, 
                                    Wn.T, 
                                    Az.T, 
                                    Au.T, 
                                    reg=admm_reg,
                                    iters=inner_iters,
                                    rho_start=rho_start))        
        
        mid = Az.norm(dim=0) + 1e-12 # normalize rows as described in 3.2
        Bz, Bu = admm(Az / mid, 
                      Wn, 
                      Bz, 
                      Bu, 
                      reg=admm_reg,
                      iters=inner_iters,
                      rho_start=rho_start)
        if itt == outer_iters - 1:
            print("err", itt, ((Az / mid).matmul(Bz) - Wn).square().sum().item(), (Wn).square().sum().item())

    # the reconstructed matrix is (Az / mid).matmul(Bz)
    # because the Bz is opted asssuming Az was normalized at the last iteration
    # so we need to apply to Az to get correct scales of values
    # print(f"relative err percent of iter {itt}: ", ((Az / mid).matmul(Bz) - Wn).square().sum().item() / (Wn).square().sum().item())
            
    return (Az / norm_o).matmul(Bz / norm), Az / norm_o, Bz / norm, 1 / mid

def factorize(lx, target_bits, admm_reg, outer_iters, inner_iters):
    W = lx.weight.detach().float()
    W2, Ac, Bc, mid = factorizef(W, 
                                 lx.i_norm, 
                                 lx.o_norm, 
                                 outer_iters=outer_iters,
                                 inner_iters=inner_iters,
                                 admm_reg=admm_reg,
                                 target_bits=target_bits)
    
    # This rescales them so that their norms are balanced. 
    # Since multiplying one matrix by alpha(Bn/An) and the other by 1/aplha leaves the product unchanged, 
    # this improves numerical stability.
    An = Ac.norm() + 1e-12
    Bn = Bc.norm() + 1e-12
    Ac *= (Bn/An).sqrt()
    Bc *= (An/Bn).sqrt()
    
    W3 = (Ac * mid).matmul(Bc)
    assert W3.shape == lx.weight.shape
    print("sparsity check: ", ((Ac != 0).sum() + (Bc != 0).sum()).item() / W3.numel(), "bpw")
    return W3, Ac, Bc, mid

def my_pack(x):
    x = (x == 1).to(torch.uint8)
    out = torch.zeros((x.shape[0]//8), device=x.device, dtype=torch.uint8)
    for i in range(8):
        out += x[i::8] << (7 - i)
    return out

# @torch.compile
def my_unpack(x):
    out = torch.zeros((x.shape[0], 8), device=x.device, dtype=torch.int8)
    for i in range(8):
        out[:,i] = (x >> (7 - i)) & 1
    return out.flatten() * 2 - 1

class BitLinear(nn.Module):
    def __init__(self, b):
        super().__init__()
        # P.S. it may good for storage, but slow down the training speed
        # b_packed = my_pack(b.flatten()) # packed binary matrix

        self.shape = b.shape
        self.register_buffer("bp", b)
        self.packed = False
    
    def pack_to_save(self):
        self.bp = my_pack(self.bp.flatten())
        self.packed = True
    
    def forward(self, x):
        if self.packed:
            return x.matmul(my_unpack(self.bp).reshape(self.shape).T.to(x.dtype))
        return x.matmul(self.bp.reshape(self.shape).T.to(x.dtype))

        
class Mul(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = torch.nn.Parameter(w, requires_grad=False)
    
    def forward(self, x):
        return x * self.w.to(x.dtype)
        

def replace(lx):
    m1 = lx.weight.B
    m2 = lx.weight.A

    # A and B matrices are solutions of ADMM with form of a * A_sign * m^T
    # so we need to redecompose it into three parts.
    u1, b1, v1 = svid_decomp(m1.float(), all_parts=True)
    u2, b2, v2 = svid_decomp(m2.float(), all_parts=True)
    
    lx2 = nn.Sequential(
        Mul(v1),
        BitLinear(b1),
        Mul(u1*lx.weight.mid*v2),
        BitLinear(b2),
        Mul(u2)
    )
    return lx2

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

# -------------------------------------------------------

def collect_norms(model, dataloader, n_calib_data=256, n_calib_limit=256, seqlen_limit=4096):
    model.cuda()
    model.gradient_checkpointing_enable()
    model.config.use_cache = False # disable KV-cache
    model.train() # turn-on gradient recording

    def f_hook(m, i, o):
        X = i[0].detach().float()
        X = X.reshape(-1, X.shape[-1])
        m.i_norm += X.square().mean(dim=0)
        
    def b_hook(m, _, go):
        X = go[0].detach().float()
        X = X.reshape(-1, X.shape[-1])
        m.o_norm += X.square().mean(dim=0) * 1e6

    # set no_grad for all linears
    for n, p in model.named_parameters():
        if "embed_tokens" not in n:
            p.requires_grad = False

    handles = []

    print("Collecting i_norm and o_norm for linear layers...")
    for n, m in model.named_modules():
        if type(m) == nn.Linear and "lm_head" not in n:
            #print(n)
            m.i_norm = torch.zeros(m.weight.shape[1], device=m.weight.device)
            m.o_norm = torch.zeros(m.weight.shape[0], device=m.weight.device)
            handles.append(m.register_forward_hook(f_hook))
            handles.append(m.register_full_backward_hook(b_hook))
    
    # from cut_cross_entropy import linear_cross_entropy
    total_len = min(n_calib_limit, n_calib_data)
    for idx, eval_batch in tqdm(enumerate(dataloader), total=total_len):
        if idx >= total_len:
            break
        # cuda_eval_batch = {"input_ids": eval_batch.cuda(), "labels": eval_batch.cuda()}
        # # cut seqlen for collecting activations (OOM for llama3-8b with seqlen=8192)
        # cuda_eval_batch["input_ids"] = cuda_eval_batch["input_ids"][:, :seqlen_limit]
        # cuda_eval_batch["labels"] = cuda_eval_batch["labels"][:, :seqlen_limit]
        # outputs = model(**cuda_eval_batch)
        # print(idx, outputs.loss)
        # outputs.loss.backward()

        # bx = eval_batch.cuda().unsqueeze(0)
        # embs = model.model(bx.cuda())[0]
        # loss = linear_cross_entropy(embs, model.lm_head.weight, bx, shift=1)
        # print(idx, loss)
        # loss.backward()
        eval_batch = eval_batch.cuda().unsqueeze(0)
        lm_logits = model(eval_batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = eval_batch[:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        print(idx, loss)
        loss.backward()

    for h in handles:
        h.remove()

    return model

def collect_activations(model, layers, dev, dataloader, n_samples):
    model.model.embed_tokens = model.model.embed_tokens.to(dev) 
    model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (n_samples, model.seqlen, model.config.hidden_size), dtype=dtype, device="cpu"
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_embeddings'] = kwargs['position_embeddings']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for idx, batch in enumerate(dataloader):
        if idx >= n_samples:
            break
        try:
            model(batch.unsqueeze(0).to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    comp_inps = inps.clone()
    attention_mask = cache['attention_mask']
    position_embeddings = cache['position_embeddings']

    return inps, comp_inps, attention_mask, position_embeddings


@torch.no_grad()
def opt_sequential(model, dataloader, dev, target_bits=2.0, n_samples=256, norm_order=2.0, 
                   n_epochs=8, lr=3e-5, weight_decay=1e-4, n_grad_accumu=8, one_cycle_lr=False,
                   admm_reg=3e-2, outer_iters=260, inner_iters=3,
                   save_dir=None, only_full_ft=False,
                   train_scaling_vectors=False, lr_s=1e-4, epoch_s=2,
                   test_dataloader=None, test_n_samples=128):

    """
    Main quantization procedure. At compression for each decoder block, we 
        1. first fine-tune all the linears (q, k, v, o, gate, up, down) to correct errors from previous blocks,
            (for the 1-st block we ignore this step)
        2. Then we compress the q, v, and o matrices, 
        3. then fine-tune the rest of the linears (k, gate, up, down) 
        3. and compress the remaining matrices (k, gate, up, down)

    """
    print('Starting layer-wise PTQ...')
    
    model.cpu()
    model.gradient_checkpointing_disable()
    model.eval()
    model.config.use_cache = False
    layers = model.model.layers

    # collecting FP inputs activation of first block for train and test dataset
    print("collecting activations...")
    inps, comp_inps, attention_mask, position_embeddings = collect_activations(model, layers, dev, dataloader, n_samples)
    if test_dataloader is not None:
        test_inps, test_comp_inps, test_attention_mask, test_position_embeddings = collect_activations(model, layers, dev, test_dataloader, test_n_samples)

    print('Ready.')
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "tb"))

    # layers = model.model.layers

    # block-wise PTQ with compression
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        subset = find_layers(layer)

        # current FP output for layer (train & test)
        print(f"collecting FP target output activations for block {i}...")
        for j in range(n_samples):
            inps[j] = layer(inps[j].unsqueeze(0).cuda(), attention_mask=attention_mask, position_embeddings=position_embeddings)[0]
        if test_dataloader is not None:
            for j in range(test_n_samples):
                test_inps[j] = layer(test_inps[j].unsqueeze(0).cuda(), attention_mask=test_attention_mask, position_embeddings=test_position_embeddings)[0]

        imp = layer.mlp.down_proj.o_norm
        
        for name in [ # the order of matrices in this list is important
            "self_attn.q_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "self_attn.k_proj",
            "mlp.up_proj",
            "mlp.gate_proj",
            "mlp.down_proj",
        ]:
            print(i, name)
            to_opt = {n: p for n, p in layer.named_parameters() if "weight" in n and "layernorm" not in n}

            # finetuning the weights layer-wise
            if only_full_ft:
                cond = ("q_proj" in name and i >= 1)
            else:
                cond = ("q_proj" in name and i >= 1) or "k_proj" in name
            if len(to_opt) > 0 and cond:
               
                err_before = 0
                for j in range(n_samples):
                    cur_out = layer(comp_inps[j].unsqueeze(0).cuda(), attention_mask=attention_mask, position_embeddings=position_embeddings)[0]
                    err_before += ((cur_out.float() - inps[j].cuda().float()).square() * imp).mean().item()

                print("err before", err_before)

                for n, p in to_opt.items():
                    p.requires_grad = True

                print("tuning weights: ", n_samples, to_opt.keys(), [v.dtype for v in to_opt.values()])
#                opt = torch.optim.AdamW(to_opt.values(), lr, weight_decay=weight_decay)
                opt = BF16FusedAdamW(to_opt.values(), lr, weight_decay=weight_decay)
                if one_cycle_lr:
                    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=n_samples*n_epochs // n_grad_accumu, cycle_momentum=False)
                else:
                    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs * n_samples // n_grad_accumu)
                print("scheduler", sch)

                # start tuning
                with torch.enable_grad():
                    for ep in range(n_epochs):
                        # train
                        err_total = 0
                        for j in range(n_samples):
                            cur_out = layer(comp_inps[j].unsqueeze(0).cuda(), attention_mask=attention_mask, position_embeddings=position_embeddings)[0]
                            err = ((cur_out.float() - inps[j].cuda().float()).abs().pow(norm_order) * imp).sum()
                            err.backward()
                            if j % n_grad_accumu == n_grad_accumu - 1:
                                opt.step()
                                sch.step()
                                layer.zero_grad(set_to_none=True)
                            err_total += err.item() / inps.shape[1] / inps.shape[2]
                        
                        # test after each epoch
                        if test_dataloader is not None:
                            with torch.no_grad():
                                test_err_total = 0
                                for j in range(test_n_samples):
                                    cur_out = layer(test_comp_inps[j].unsqueeze(0).cuda(), attention_mask=test_attention_mask, position_embeddings=test_position_embeddings)[0]
                                    err = ((cur_out.float() - test_inps[j].cuda().float()).abs().pow(norm_order) * imp).sum()
                                    test_err_total += err.item() / test_inps.shape[1] / test_inps.shape[2]

                        if len(to_opt) == 4:
                            writer.add_scalar(f"train_loss.layer{i}/train_part_tune_k_gate_up_down", err_total, n_samples * ep)
                            if test_dataloader is not None:
                                writer.add_scalar(f"train_loss.layer{i}/test_part_tune_k_gate_up_down", test_err_total, test_n_samples * ep)
                        else:
                            writer.add_scalar(f"train_loss.layer{i}/train_all_tune", err_total, n_samples * ep)
                            if test_dataloader is not None:
                                writer.add_scalar(f"train_loss.layer{i}/test_all_tune", test_err_total, test_n_samples * ep)
                        print(f"epoch: {ep}, train err: {err_total}")
                        
            # compress matrices
            print(f'Factorizing {i}.{name}...')
            lx = subset[name]
            W2, Ac, Bb, mid, = factorize(lx, 
                                         target_bits=target_bits, 
                                         admm_reg=admm_reg,
                                         outer_iters=outer_iters,
                                         inner_iters=inner_iters)
            W2 = W2.T
            fact_err = (W2.T - lx.weight).square().sum().item()
            base_err =  lx.weight.square().sum().item()
            print("relative err after factorization: ", fact_err/base_err, fact_err, "\n")
            
            # assign compressed matrices to current layer
            lx.weight.data = W2.T.to(lx.weight)
            lx.weight.A = Ac
            lx.weight.B = Bb
            lx.weight.mid = mid
            parts = name.split('.')
            block = getattr(layer, parts[0])
            setattr(block, parts[1], replace(lx)) # after replacement, the pruned linear will not contain 'weight' in name, and will not be opt.

        if train_scaling_vectors:
            sv_to_opt = {n: p for n, p in layer.named_parameters() if (".0.w" in n or ".2.w" in n or ".4.w" in n) and "layernorm" not in n}
            print("tuning scaling vectors: ", sv_to_opt.keys())
            for n, p in sv_to_opt.items():
                print(n)
                p.requires_grad = True
            sv_opt = torch.optim.AdamW(sv_to_opt.values(), lr=lr_s, weight_decay=weight_decay)
            sv_sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epoch_s * n_samples // n_grad_accumu)
            with torch.enable_grad():
                for ep in range(epoch_s):
                    err_total = 0
                    for j in range(n_samples):
                        cur_out = layer(comp_inps[j].unsqueeze(0).cuda(), attention_mask=attention_mask, position_embeddings=position_embeddings)[0]
                        err = ((cur_out.float() - inps[j].cuda().float()).abs().pow(norm_order) * imp).sum()
                        err.backward()
                        if j % n_grad_accumu == n_grad_accumu - 1:
                            sv_opt.step()
                            sv_sch.step()
                            layer.zero_grad(set_to_none=True)
                        err_total += err.item() / inps.shape[1] / inps.shape[2]
                    print(f"epoch: {ep}, train err: {err_total}")
                    writer.add_scalar(f"train_loss.layer{i}/train_scaling_vectors", err_total, n_samples * ep)

            
        # collect the output of compressed block (input for next block) (train & test)
        if i != (len(layers) - 1):
            print(f"collecting Q input activations for next block ...")
            for j in range(n_samples):
                comp_inps[j] = layer(comp_inps[j].unsqueeze(0).cuda(), attention_mask=attention_mask, position_embeddings=position_embeddings)[0]
            if test_dataloader is not None:
                for j in range(test_n_samples):
                    test_comp_inps[j] = layer(test_comp_inps[j].unsqueeze(0).cuda(), attention_mask=test_attention_mask, position_embeddings=test_position_embeddings)[0]
            
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

@torch.no_grad()
def eval_model(model, eval_ids):
    eval_ids = eval_ids.input_ids.flatten()
    eval_ids = eval_ids[:len(eval_ids)//model.seqlen*model.seqlen]
    eval_dataloader = eval_ids.reshape(-1,1,model.seqlen)
    model.eval()

    print("\nStart evaluation...", len(eval_dataloader), len(eval_ids))
    task_loss = 0
    n_processed_seq = 0
    for _, eval_batch in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
        n_batch_seq = eval_batch.shape[0]

        cuda_eval_batch = {"input_ids": eval_batch.cuda(), "labels": eval_batch.cuda()}

        with torch.no_grad():
            outputs = model(**cuda_eval_batch)

        batch_loss = outputs.loss.item()
        task_loss *= n_processed_seq / (n_processed_seq + n_batch_seq)
        task_loss += batch_loss * n_batch_seq / (n_processed_seq + n_batch_seq)
        n_processed_seq += n_batch_seq
        ppl = math.exp(task_loss)

    return ppl
