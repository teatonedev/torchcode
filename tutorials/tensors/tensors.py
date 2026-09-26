import torch
import numpy as np
# they are quite similar and share many similarities ndarray (np) and tensors (torch)

# Tensors can be defined or initialized as following examples;

# data -> tensor
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

print(x_data)


# np array -> tensor
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

x_ones = torch.ones_like(x_data)
print(f" Ones tensor: \n\n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"random tensor: \n {x_rand} \n ")


# random or contants
shape = (2,3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f" RAND tensor: \n {rand_tensor} \n")
print(f" ONES tensor: \n {ones_tensor} \n")
print(f" ZEROS tensor: \n {zeros_tensor} \n")



#ATTRBTS

tensor = torch.rand(3,4)
print(f"\n\nMy tensor: {tensor}")
print("specs of my tensor")
print(f"shape of tensor: {tensor.shape}")
print(f"datatype of tensor: {tensor.dtype}")
print(f"device tensor on: {tensor.device}")



# they are created on CPU by default.

if torch.accelerator.is_available():
    dev = torch.accelerator.current_accelerator()
    print(f"\n\nCURRENT ACCLRTR: {dev}")
    tensor = tensor.to(dev)
    print(tensor)


# indexing and slicing

tensor = torch.ones(4,4)
print("INITIAL\n", tensor)

print(f"first row: {tensor[0]}")
print(f"first column: {tensor[:, 0]}")
print(f"last column: {tensor[..., -1]}")
tensor[:,1] = 0
print("AFTER [:,1]", tensor)

tensor[:,3] = -31
print("AFTER [:,3]", tensor)


#join tensors with cat <3

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)


#arithmetic

y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)

z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
print(z3)


# single el tensors 1 vall
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))


# dont use dangerous
print(f"{tensor} \n ")
tensor.add_(31)
print(tensor)


# bridge with NP 
# tensors CPU and np arrs share memory

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# lets change t and see what happens on n as well.
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")


# np arr -> tensor

n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")

