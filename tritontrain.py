import torch
import triton
import triton.language as tl
import time
import torch
import torch.nn as nn
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import math

@triton.jit()
def tile_schedule(pid,
                  NUM_SM: tl.constexpr, total_tiles: tl.constexpr):

    start = (pid*total_tiles) // NUM_SM
    end = (((pid+1)*total_tiles) // NUM_SM)

    return start, end


@triton.jit()
def gemm_balanced(a_ptr, b_ptr, c_ptr,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            prob_m, prob_n, prob_k,
            block_m: tl.constexpr,
            block_n: tl.constexpr,
            block_k: tl.constexpr,
            NUM_SM: tl.constexpr,
            total_tiles: tl.constexpr,
            ):

    pid = tl.program_id(0)

    num_n_tiles = tl.cdiv(prob_n, block_n)
    num_k_tiles = tl.cdiv(prob_k, block_k)

    start, end = tile_schedule(pid, NUM_SM, total_tiles)
    for tile_id in range(start, end):

        tile_m_idx = tile_id // num_n_tiles
        tile_n_idx = tile_id % num_n_tiles

        offs_m = tile_m_idx*block_m + tl.arange(0, block_m)
        offs_n = tile_n_idx*block_n + tl.arange(0, block_n)
        offs_k = tl.arange(0, block_k)

        # Compiler Hint for Vectorized Load
        offs_am = tl.max_contiguous(tl.multiple_of(offs_m, block_m), block_m)
        offs_bn = tl.max_contiguous(tl.multiple_of(offs_n, block_n), block_n)

        a_ptrs = a_ptr + (offs_am[:, None]*stride_am + offs_k[None, :]*stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None]*stride_bk + offs_bn[None, :]*stride_bn)

        acc = tl.zeros([block_m, block_n], tl.float32)
        for kk in range(0, num_k_tiles):

            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)

            acc = tl.dot(a, b, acc, out_dtype=tl.float32)

            a_ptrs += block_k*stride_ak
            b_ptrs += block_k*stride_bk

        acc.to(tl.bfloat16)

        offs_cm = tile_m_idx*block_m + tl.arange(0, block_m)
        offs_cn = tile_n_idx*block_n + tl.arange(0, block_n)

        c_ptrs = c_ptr + stride_cm*offs_cm[:, None] + stride_cn*offs_cn[None, :]
        tl.store(c_ptrs, acc)


@triton.jit
def gemm_grouped_matmul_kernel(
    # device tensor of matrices pointers
    group_a_ptrs,
    group_b_ptrs,
    group_c_ptrs,
    # device tensor of gemm sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <M, N, K> of each gemm
    group_gemm_sizes,
    # device tensor of leading dimension sizes. its shape is [group_size, 3]
    # dim 0 is group_size, dim 1 is the values of <lda, ldb, ldc> of each gemm
    g_lds,
    # number of gemms
    group_size,
    # number of virtual SM
    NUM_SM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # breakpoint()
    tile_idx = tl.program_id(0)
    last_problem_end = 0
    for g in range(group_size):
        # get the gemm size of the current problem
        gm = tl.load(group_gemm_sizes + g * 3)
        gn = tl.load(group_gemm_sizes + g * 3 + 1)
        gk = tl.load(group_gemm_sizes + g * 3 + 2)
        num_m_tiles = tl.cdiv(gm, BLOCK_SIZE_M)
        num_n_tiles = tl.cdiv(gn, BLOCK_SIZE_N)
        num_tiles = num_m_tiles * num_n_tiles
        # iterate through the tiles in the current gemm problem
        while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):
            # pick up a tile from the current gemm problem
            k = gk
            lda = tl.load(g_lds + g * 3)
            ldb = tl.load(g_lds + g * 3 + 1)
            ldc = tl.load(g_lds + g * 3 + 2)
            a_ptr = tl.load(group_a_ptrs + g).to(tl.pointer_type(tl.bfloat16))
            b_ptr = tl.load(group_b_ptrs + g).to(tl.pointer_type(tl.bfloat16))
            c_ptr = tl.load(group_c_ptrs + g).to(tl.pointer_type(tl.bfloat16))
            # figure out tile coordinates
            tile_idx_in_gemm = tile_idx - last_problem_end
            tile_m_idx = tile_idx_in_gemm // num_n_tiles
            tile_n_idx = tile_idx_in_gemm % num_n_tiles

            # do regular gemm here
            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs_k = tl.arange(0, BLOCK_SIZE_K)
            a_ptrs = a_ptr + offs_am[:, None] * lda + offs_k[None, :]
            b_ptrs = b_ptr + offs_k[:, None] * ldb + offs_bn[None, :]
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for kk in range(0, tl.cdiv(k, BLOCK_SIZE_K)):

                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)

                accumulator += tl.dot(a, b)
                a_ptrs += BLOCK_SIZE_K
                b_ptrs += BLOCK_SIZE_K * ldb

            c = accumulator.to(tl.bfloat16)

            offs_cm = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + ldc * offs_cm[:, None] + offs_cn[None, :]

            # assumes full tile for now
            tl.store(c_ptrs, c)

            # go to the next tile by advancing NUM_SM
            tile_idx += NUM_SM

        # get ready to go to the next gemm problem
        last_problem_end = last_problem_end + num_tiles


def group_gemm_fn(group_A, group_B, config = None):
    device = 'cuda'
    assert len(group_A) == len(group_B)
    group_size = len(group_A)

    A_addrs = []
    B_addrs = []
    C_addrs = []
    g_sizes = []
    g_lds = []
    group_C = []
    for i in range(group_size):
        A = group_A[i]
        B = group_B[i]
        assert A.shape[1] == B.shape[0]
        M, K = A.shape
        K, N = B.shape
        C = torch.empty((M, N), device=device, dtype=A.dtype)
        group_C.append(C)
        A_addrs.append(A.data_ptr())
        B_addrs.append(B.data_ptr())
        C_addrs.append(C.data_ptr())
        g_sizes += [M, N, K]
        g_lds += [A.stride(0), B.stride(0), C.stride(0)]

    # note these are device tensors
    d_a_ptrs = torch.tensor(A_addrs, device=device)
    d_b_ptrs = torch.tensor(B_addrs, device=device)
    d_c_ptrs = torch.tensor(C_addrs, device=device)
    d_g_sizes = torch.tensor(g_sizes, dtype=torch.int32, device=device)
    d_g_lds = torch.tensor(g_lds, dtype=torch.int32, device=device)
    # we use a fixed number of CTA, and it's auto-tunable

    num_sm = 132
    if config:
        block_m = config["block_m"]
        block_n = config["block_n"]
        block_k = config["block_k"]
        num_warps = config["num_warps"]
        num_stages = config["num_stages"]

    else:
        block_m = 128
        block_n = 256
        block_k = 32
        num_warps = 8
        num_stages = 4

    grid = (num_sm, )
    gemm_grouped_matmul_kernel[grid](
        d_a_ptrs,
        d_b_ptrs,
        d_c_ptrs,
        d_g_sizes,
        d_g_lds,
        group_size,
        NUM_SM=num_sm,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return group_C



class _matmul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, a, b):

        m, _ = a.shape
        k, n = b.shape

        block_m = 128
        block_n = 256
        block_k = 64
        num_warps = 8
        num_stages = 2
        sms = 132

        num_m_tiles = triton.cdiv(m, block_m)
        num_n_tiles = triton.cdiv(n, block_n)
        total_tiles = num_m_tiles * num_n_tiles

        c = torch.zeros(m, n, dtype=torch.bfloat16, device=a.device)
        grid = (sms,)

        gemm_balanced[grid](a, b, c,
                    a.stride(0), a.stride(1),
                    b.stride(0), b.stride(1),
                    c.stride(0), c.stride(1),
                    m, n, k,
                    block_m=block_m, block_n=block_n, block_k=block_k,
                    NUM_SM=sms, total_tiles=total_tiles, num_warps=num_warps,
                    num_stages=num_stages)

        ctx.save_for_backward(a, b)
        return c

    @staticmethod
    def backward(ctx, dL_dc):
        """
        Equations:
                 1. dL/da = dL/dc @ b.T
                 2. dL/db = a.T @ dL/dc
        Shapes:
                GEMM1:
                dL/dc: (m, n)
                b.T:   (n, k)
                ==> dL/da: (m, k)

                GEMM2:
                a.T:    (k, m)
                dL/dc:  (m, n)
                ==> dL/db: (k, m)
        Notation:
                dL_da: partial(L)/partial(a)
                dL_db: partial(L)/partial(b)
                dL_dc: partial(L)/partial(c)
        """
        a, b = ctx.saved_tensors

        group_a = [dL_dc, a.permute(1, 0).contiguous()]
        group_b = [b.permute(1, 0).contiguous(), dL_dc]

        group_derivs = group_gemm_fn(group_a, group_b)

        return group_derivs[0], group_derivs[1]

matmul = _matmul.apply


# Define the custom linear layer
class TritonLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(TritonLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.bfloat16, device='cuda'))


    def forward(self, input):

        return matmul(input, self.weight.T)

class CustomNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(CustomNet, self).__init__()
        self.linear = TritonLinear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

class BaseNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(BaseNet, self).__init__()
        self.linear = nn.Linear(input_size, output_size, dtype=torch.bfloat16)

    def forward(self, x):
        return self.linear(x)

batch_size =  4096      # M
input_size =  4096      # K
output_size = 4096      # N
num_samples = 16384


torch.cuda.manual_seed(3227)
# Generate random input data and labels
inputs = torch.randn(num_samples, input_size, dtype=torch.bfloat16, device='cuda')
labels = torch.randint(0, output_size, (num_samples,), dtype=torch.long, device='cuda')

# Create a DataLoader
dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Function to train a model and record loss
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    loss_history = []
    model = model.to('cuda')
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            # print(f"{inputs.shape=}")
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(f"{outputs.shape=}")
            # print(f"{outputs=}")
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        average_loss = epoch_loss / len(dataloader)
        loss_history.append(average_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}")
    return loss_history


# Function to train a model and record loss with timing
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    loss_history = []
    model = model.to('cuda')

    warmup_steps = 5
    # Warmup phase
    print("Starting warmup phase...")
    warmup_start_time = time.time()
    for i, (inputs, labels) in enumerate(dataloader):
        if i >= warmup_steps:
            break
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        outputs = model(inputs)
        # Perform a dummy loss calculation and backward pass to warm up the kernel
        loss = criterion(outputs, labels)
        loss.backward()
    warmup_end_time = time.time()
    print(f"Warmup phase completed in {warmup_end_time - warmup_start_time:.2f}s")

    # Start timing the training process
    total_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()  # Start timing for the epoch
        epoch_loss = 0.0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            # print(f"{inputs.shape=}")
            optimizer.zero_grad()
            outputs = model(inputs)
            # print(f"{outputs.shape=}")
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        average_loss = epoch_loss / len(dataloader)
        loss_history.append(average_loss)

        epoch_end_time = time.time()  # End timing for the epoch
        epoch_time = epoch_end_time - epoch_start_time
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}, Time: {epoch_time:.2f}s")

    total_end_time = time.time()  # End timing the training process
    total_training_time = total_end_time - total_start_time
    print(f"Total training time: {total_training_time:.2f}s")

    return loss_history

# Instantiate models, loss function, and optimizers
custom_model = CustomNet(input_size, output_size)
base_model = BaseNet(input_size, output_size)

criterion = nn.CrossEntropyLoss()

custom_optimizer = optim.Adam(custom_model.parameters(), lr=0.001)
base_optimizer = optim.Adam(base_model.parameters(), lr=0.001)


print("Training Triton Model 1 Layer MLP:")
custom_loss_history = train_model(custom_model, dataloader, criterion, custom_optimizer)


print("Training PyTorch Model 1 Layer MLP:")
custom_loss_history = train_model(base_model, dataloader, criterion, custom_optimizer)
