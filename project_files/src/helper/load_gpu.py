import torch

def load_gpu():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device = torch.device("cuda")
        if "AMD" in device_name or "Radeon" in device_name:
            print(f"Verwende AMD ROCm GPU: {device_name}")
        else:
            print(f"Verwende CUDA GPU: {device_name}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Verwende MPS GPU (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Verwende CPU - (Keine GPU erkannt)")

    return device