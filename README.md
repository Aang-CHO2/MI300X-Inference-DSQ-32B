# DeepSeek-Qwen ROCm Chat & Batch Inference
Talk to a powerful LLM on an AMD MI300X GPU—right from your terminal!

🚦 **What’s This Project?**
This is a command-line tool that lets you chat with the DeepSeek-R1-Distill-Qwen-32B large language model, or send it a batch of prompts and get answers back. It’s built for AMD GPUs using ROCm, and it even shows you live GPU stats while you use it!

💡 🖥️ About the AMD MI300X Virtualization
Supercharged Hardware:
We’re running on an AMD Instinct MI300X GPU, using AMD’s [AMD’s open-source GIM driver](https://github.com/amd/MxGPU-Virtualization) with SR-IOV technology. That means we can split the GPU into multiple “virtual GPUs” (VFs) and share them between different virtual machines (VMs). In this setup, one VM gets 2 out of 8 possible VFs—so you get a big slice of a supercomputer, but other workloads can use the rest! Here’s the summar: 
- The MI300X GPU can be split into 8 “virtual GPUs” (VFs).
- Our VM is using 2 of those VFs, so it gets a big chunk of the GPU’s power.
- This setup is awesome for sharing one GPU between different users or workloads, without sacrificing speed or security.

In short: the MI300X is a powerhouse, and SR-IOV lets us use it flexibly and efficiently!

**Script capabilities:** 
- Flexible: Chat with the model interactively, or send it a bunch of prompts at once.
- Live GPU Dashboard:
- See GPU’s memory, temperature, and usage in real time.
- Easy to Use: Just run the script, pick a mode, and start chatting or batch prompting.

**Prerequisites**
- Host OS: Ubuntu 22.04
- Virtual Machine OS: Ubuntu 22.04
- ROCm Version: [ROCm 6.4](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) installed inside the VM
- AMD MI300X GPU with [GIM driver](https://github.com/amd/MxGPU-Virtualization/releases) installed 
- ROCm-enabled PyTorch (Install Guide)
- Transformers library (pip install transformers)
- DeepSeek-R1-Distill-Qwen-32B model downloaded locally from [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)

