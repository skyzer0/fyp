import torch
import torchvision

# Print PyTorch version
print("PyTorch version:", torch.__version__)

# Print torchvision version
print("Torchvision version:", torchvision.__version__)

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available!")
    print("CUDA version:", torch.version.cuda)
    print("GPU Device Name:", torch.cuda.get_device_name())
else:
    print("CUDA is not available. Using CPU.")
