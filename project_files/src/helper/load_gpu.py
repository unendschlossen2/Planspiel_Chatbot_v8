import torch

def load_gpu():
    """
    Detects and returns the available processing device (CUDA, MPS, or CPU)
    and a formatted string with the device name.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        # Check for AMD/Radeon to correctly label ROCm, otherwise assume CUDA
        if "AMD" in device_name.upper() or "RADEON" in device_name.upper():
            display_name = f"ROCm: {device_name}"
        else:
            display_name = f"CUDA: {device_name}"
        print(f"Using GPU: {display_name}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        display_name = "MPS (Apple Silicon)"
        print(f"Using GPU: {display_name}")
    else:
        device = torch.device("cpu")
        display_name = "CPU"
        print("No GPU detected, using CPU.")

    return device, display_name