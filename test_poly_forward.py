# test the PolyProbe model's factorized forward pass matches the expected output
from model import PolyProbe

import torch
from einops import einsum
import tensorly as tl  

tl.set_backend('pytorch')
torch.manual_seed(42)

test_in_features = 4
test_out_features = 1
test_rank = 4

# SYMMETRIC
# 4th order poly test
with torch.no_grad():

    # symmetric
    test_model = PolyProbe(in_features=test_in_features, out_features=test_out_features, max_order=4, ranks=[test_rank]*3)
    
    test_z = torch.randn(1, test_in_features)*10 # scale up to ensure no possible underflow etc

    # output from model
    y_hat = test_model(test_z, test_time_order=4)
    
    partial_outputs = []

    ############## 1st order term
    y1 = test_model.W[0] + einsum(test_z, test_model.W[1], '... i, o i -> ... o')
    partial_outputs.append(y1)

    ############## 2nd order term
    W = einsum(test_model.lam[0], test_model.HO[0], test_model.HO[0], 'r, r i1, r i2 -> i1 i2') # explicitly construct tensor with einsum
    W = tl.cp_tensor.cp_to_tensor((test_model.lam[0], [test_model.HO[0].T for o in range(2)]))  # or: form tensor with tensorly
    
    y2 = einsum(test_z, test_z, W, '... i1, ... i2, i1 i2 -> ...').unsqueeze(-1)
    partial_outputs.append(y1+y2)

    ############## 3rd order term
    W = einsum(test_model.lam[1], test_model.HO[1], test_model.HO[1], test_model.HO[1], 'r, r i1, r i2, r i3 -> i1 i2 i3') # explicitly construct tensor with einsum
    W = tl.cp_tensor.cp_to_tensor((test_model.lam[1], [test_model.HO[1].T for o in range(3)]))  # or: form tensor with tensorly
    
    y3 = einsum(test_z, test_z, test_z, W, '... i1, ... i2, ... i3, i1 i2 i3 -> ...').unsqueeze(-1)
    partial_outputs.append(y1+y2+y3)

    ############## 4th order term
    W = einsum(test_model.lam[2], test_model.HO[2], test_model.HO[2], test_model.HO[2], test_model.HO[2], 'r, r i1, r i2, r i3, r i4 -> i1 i2 i3 i4') # explicitly construct tensor with einsum
    W = tl.cp_tensor.cp_to_tensor((test_model.lam[2], [test_model.HO[2].T for o in range(4)]))  # or: form tensor with tensorly
    
    y4 = einsum(test_z, test_z, test_z, test_z, W, '... i1, ... i2, ... i3, ... i4, i1 i2 i3 i4 -> ...').unsqueeze(-1)
    partial_outputs.append(y1+y2+y3+y4)
    
    # assert each truncation equal
    for oi, (out_expected, out_actual) in enumerate(zip(y_hat, partial_outputs)):
        assert torch.allclose(out_expected, out_actual, atol=1e-6), "Outputs do not match expected values"
        print(f"Passed: output at order {oi} matches expected value. Difference: {(out_expected - out_actual).abs().max().item()}")