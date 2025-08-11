import torch

def check_gpu():
    """
    Checks for CUDA availability and prints GPU information.
    """
    print(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"Is CUDA available? {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version used by PyTorch: {torch.version.cuda}")
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs available: {gpu_count}")
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. PyTorch is using the CPU.")
        print("Please ensure you have a CUDA-enabled version of PyTorch installed and that your NVIDIA drivers are up to date.")

if __name__ == "__main__":
    check_gpu()