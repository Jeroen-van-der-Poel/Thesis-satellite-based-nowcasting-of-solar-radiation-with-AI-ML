import torch
print(torch.cuda.is_available())         # should be True
print(torch.version.cuda)                # should be '12.1'
print(torch.cuda.get_device_name(0))     # should show "NVIDIA RTX A6000"
