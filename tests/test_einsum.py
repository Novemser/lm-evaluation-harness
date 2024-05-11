import torch
from tqdm import tqdm
from transformers.precomputation import generate_act_hats_llama2

def generate_act_hats_llama2_(
    d_model: int, 
    d_intermediate: int, 
    record_act_result: torch.Tensor,
    up_proj: torch.Tensor, 
    down_proj: torch.Tensor):
    # Reshape record_act_result for batch processing
    record_act_result = record_act_result.reshape(-1, d_intermediate)
    num_labels = record_act_result.shape[0]
    print(f"num_labels: {num_labels}")
    
    # Pre-compute the denominator for all i, j (this is a batch operation)
    # The dimensions need to be aligned: 'ik,kj->ij' where i is for d_model and k is for d_intermediate
    denominator = torch.einsum('ik,kj->ij', [down_proj, up_proj])
    
    # Compute the numerator using einsum for batch processing
    # Align dimensions: 'lk,ik,kj->lj' where l is for num_labels, i is for d_model, k is for d_intermediate
    numerator = torch.einsum('lk,ik,kj->lj', [record_act_result, down_proj, up_proj])
    
    # Compute a_hat using broadcasting
    a_hat = numerator / denominator.unsqueeze(0)
    
    # No need to concatenate tensors, as they are already in batch form
    return a_hat.reshape(num_labels, d_model, d_model)

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
output = generate_act_hats_llama2(d_model, d_intermediate, record_act_result, up_proj, down_proj)
print(output)  # Should be (num_labels, d_model, d_model)