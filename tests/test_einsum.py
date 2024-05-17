import torch
from tqdm import tqdm
from transformers.precomputation import generate_act_hats_llama2

# Dummy data for demonstration purposes
d_model = 3
d_intermediate = 4
num_labels = 1
# record_act_result = torch.rand(num_labels, d_intermediate)
up_proj = torch.rand(d_intermediate, d_model)
down_proj = torch.rand(d_model, d_intermediate)
record_act_result = torch.tensor([0.01, 0.02, 0.03, 0.04])
for i in range(d_intermediate):
    for j in range(d_model):
        up_proj[i][j] = 1
        down_proj[j][i] = 2
print("record_act_result:", record_act_result)
print("up_proj:", up_proj)
print("down_proj:", down_proj)
# Call the function with dummy data
generate_act_hats_llama2(d_model, d_intermediate, record_act_result, up_proj.transpose(0, 1), down_proj.transpose(0, 1), save_result=False)