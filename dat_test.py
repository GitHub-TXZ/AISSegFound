from dat import DAT
import torch
# Generate a fake tensor for testing
fake_input = torch.randn(1, 3, 224, 224)  # Example: Batch size 1, 3 channels, 224x224 image

dat = DAT()

# Test the model with the fake tensor
output = dat(fake_input)
print(output)
