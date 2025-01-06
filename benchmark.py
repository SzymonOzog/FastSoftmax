import torch
import triton
import triton.language as tl
from triton.runtime import driver
from torch.utils.cpp_extension import load
from fine_tune import fine_tune_kernel, time_kernel_ncu
torch.set_default_device('cuda')

def tprint(*args, logfile='out.log', **kwargs):
    # Print to stdout
    print(*args, **kwargs)
    
    # Append to file
    with open(logfile, 'a') as f:
        print(*args, file=f, **kwargs)
    
    # Force flush stdout
    import sys
    sys.stdout.flush()

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

device = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(device)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}
# https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
def softmax_triton(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    num_stages = 4 if SIZE_SMEM > 200000 else 1
    y = torch.empty_like(x)
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                       num_stages=num_stages, num_warps=num_warps, grid=(1, ))
        kernel._init_handles()
        n_regs = kernel.n_regs
        size_smem = kernel.metadata.shared
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy
        kernels[BLOCK_SIZE] = (kernel, num_programs)

    num_programs = min(num_programs, n_rows)

    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols,)
    return y

init_triton='''
import torch
import triton
import triton.language as tl
from triton.runtime import driver
from torch.utils.cpp_extension import load
from fine_tune import fine_tune_kernel
torch.set_default_device('cuda')


@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

device = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(device)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}
# https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
def softmax_triton(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    num_stages = 4 if SIZE_SMEM > 200000 else 1
    y = torch.empty_like(x)
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                       num_stages=num_stages, num_warps=num_warps, grid=(1, ))
        kernel._init_handles()
        n_regs = kernel.n_regs
        size_smem = kernel.metadata.shared
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy
        kernels[BLOCK_SIZE] = (kernel, num_programs)

    num_programs = min(num_programs, n_rows)

    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols,)
    return y
'''

init_torch = '''
import torch
torch.set_default_device('cuda')
'''

if __name__ == "__main__":
    times_triton = []
    times_torch = []
    times_cuda = [[] for i in range(11)]

    for pow in range(10, 18):
        x = torch.rand(128, 2**pow, device='cuda')
        y = torch.softmax(x, dim=-1)
        y2 = softmax_triton(x)
          
        torch_fn = init_torch + f'''
x = torch.rand(128, 2**{pow}, device='cuda')
out = torch.softmax(x, dim=-1)'''
        torch_ms = time_kernel_ncu(torch_fn)
        times_torch.append(torch_ms)
        tprint(f"torch took {torch_ms:.4f}")
        triton_fn = init_triton + f'''
x = torch.rand(128, 2**{pow}, device='cuda')
out = softmax_triton(x)'''
        triton_ms = time_kernel_ncu(triton_fn)
        times_triton.append(triton_ms)
        tprint(f"triton took {triton_ms:.4f}")

        for variant in range(3, 10):
            (dim_y, unroll), time = fine_tune_kernel(variant, variant > 6, x, pow, variant>4)
            times_cuda[variant].append(time)
            cuda = load(name='softmax_cuda', sources=["interface.cpp", "kernels.cu"], verbose=False, extra_cuda_cflags=[f"-lineinfo", "--use_fast_math", "-O3", f"-DSOFTMAX_VARIANT={variant}", f"-DBLOCK_DIM_Y={dim_y}", f"-DUNROLL_FACTOR={unroll}", f"-DWIDTH={2**pow}"], extra_cflags=[f"-DSOFTMAX_VARIANT={variant}", f"-DBLOCK_DIM_Y={dim_y}", f"-DUNROLL_FACTOR={unroll}", f"-DWIDTH={2**pow}"])
            y3 = cuda.softmax_cuda(x)

            assert torch.allclose(y, y2, atol=1e-8, rtol=1e-8), (y, y2)
            assert torch.allclose(y, y3, atol=1e-8, rtol=1e-8), (y, y3)
            tprint(f"cuda took {times_cuda[variant][-1]:.4f}")
    tprint("times torch", times_torch)
    tprint("times triton", times_triton)
    tprint("times cuda", times_cuda)

