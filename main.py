import numpy as np
import time
import torch

x = torch.Tensor(5, 3)

print(x)

x_ones = torch.ones(5,3)
print(x_ones)

x_zeros = torch.zeros(5,3)
print(x_zeros)

x_uniform = torch.rand(5,3)
print(x_uniform)

np_array = np.array([1., 2., 3.])
print(np_array)
torch_tensor = torch.from_numpy(np_array)
print(torch_tensor)

# Modify the Tensor
torch_tensor[0] = -1.0
print(np_array)

another_torch_tensor = torch.rand(3)
print(another_torch_tensor)
another_np_array = another_torch_tensor.numpy()
print(another_np_array)

# Modify ndarray
another_np_array[0] *= 2.0
print(another_torch_tensor)

print("GPU Accleration...")

print("Baseline")
mat_cpu = torch.rand(5000, 5000)
cpu_start = time.time()
print(torch.mm(mat_cpu.t(), mat_cpu))
cpu_end = time.time()
print("Timer[CPU]:" + str(cpu_end - cpu_start))

if torch.cuda.is_available():
    print("cuda is available!")
    mat_gpu = torch.rand(5000, 5000).cuda()
    gpu_start = time.time()
    print(torch.mm(mat_gpu.t(), mat_gpu))
    gpu_end = time.time()
    print("Timer[GPU]:" + str(gpu_end - gpu_start))
else:
   print("cuda is NOT available")
