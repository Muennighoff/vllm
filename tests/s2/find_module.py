import torch
for p in torch.ops.loaded_libraries:
    print(p)
