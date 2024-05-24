import torch
from tqdm import tqdm
from transformers.precomputation import generate_x_hats_llama2
import random
import numpy as np

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

# Dummy data for demonstration purposes
d_model = 3
d_intermediate = 4
num_labels = 1
# record_act_result = torch.rand(num_labels, d_intermediate)
gate_proj = torch.rand(d_model, d_intermediate).float().cuda()
up_proj = torch.rand(d_model, d_intermediate).float().cuda()
down_proj = torch.rand(d_intermediate, d_model).float().cuda()
record_inputs = torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.3, 0.25]]).float().cuda()
# for i in range(d_intermediate):
#     for j in range(d_model):
#         up_proj[i][j] = 1
#         down_proj[j][i] = 2
# gate_proj=up_proj * 0.6
print("gate_proj:", gate_proj)
print("up_proj:", up_proj)
print("down_proj:", down_proj)
# Call the function with dummy data
generate_x_hats_llama2(d_model=d_model, d_intermediate=d_intermediate, gate_proj=gate_proj, up_proj=up_proj, down_proj=down_proj, save_result=False, record_inputs=record_inputs, batch_size=1)