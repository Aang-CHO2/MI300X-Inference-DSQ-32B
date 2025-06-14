# Supercharged LLM Inference on AMD MI300X with Virtualization  
*A Deep Dive into MI300X-Inference-DSQ-32B*

The world of large language models (LLMs) is moving fast, and so is the need for hardware and software that can keep up. The [MI300X-Inference-DSQ-32B](https://github.com/dumroo/MI300X-Inference-DSQ-32B) project is a fantastic example of how to run a massive 32-billion parameter LLM on AMD’s MI300X GPU, all inside a virtual machine. In this post, I’ll walk you through what this project does, how it works, and how you can get started with your own high-performance, virtualized LLM inference.

---

## 🚀 What’s This Project All About?

At its core, MI300X-Inference-DSQ-32B is a command-line tool that lets you interact with the [DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) model. You can chat with the model in real time or run it in batch mode to process lots of prompts at once. The magic happens on a virtual machine powered by AMD’s MI300X GPU, split up using SR-IOV virtualization so multiple users or tasks can share the same physical hardware.

**Key Ingredients:**
- **AMD MI300X GPU:** 192GB HBM3, 304 compute units, 5.3TB/s bandwidth—seriously powerful!
- **SR-IOV Virtualization:** Carve up the GPU into up to 8 virtual slices (VFs) for multi-user or multi-VM setups.
- **ROCm 6.4:** AMD’s open-source compute stack for GPU acceleration.
- **PyTorch & Hugging Face Transformers:** For easy, flexible LLM inference.

---

## 🛠️ How Do You Set It Up?

**What You’ll Need:**
- Ubuntu 22.04 VM with 2 out of 8 VFs from an MI300X GPU (thanks to GIM/SR-IOV)
- ROCm 6.4, ROCm-enabled PyTorch, and Transformers
- The DeepSeek-R1-Distill-Qwen-32B model from Hugging Face

**Quick Setup Steps:**
1. **Install ROCm 6.4** and the GIM driver on your host machine.
2. **Assign 2 VFs** to your VM using SR-IOV.
3. Inside the VM, install **PyTorch** and **Transformers** (make sure they’re ROCm-enabled).
4. Clone the repo and download the model weights.

---

## 🧐 A Peek Inside the Inference Script

The heart of the project is [`DSQ-32B_Inference.py`](https://github.com/dumroo/MI300X-Inference-DSQ-32B/blob/main/DSQ-32B_Inference.py). Here’s what makes it tick:

### 1. **Flexible Modes**

You can run the script in two ways:
- **Chat Mode:** Talk to the LLM in real time, right from your terminal.
- **Batch Mode:** Feed it a file of prompts, and it spits out a file of completions.


### 2. **Efficient Model Loading**

The script loads the DeepSeek-R1-Distill-Qwen-32B model and tokenizer using Hugging Face Transformers, and runs them on the GPU using ROCm:


### 3. **Live GPU Monitoring**

One of the coolest features: real-time GPU stats! The script uses `rocm-smi` to show you memory usage, temperature, and utilization as you run your inference jobs. It’s like having a dashboard for your GPU right in your terminal.


### 4. **Simple Inference Loop**

- **Chat Mode:** Type your prompt, get a response.
- **Batch Mode:** The script reads prompts from a file, generates completions, and writes them to another file.


---

## 🖥️ Why Virtualization Matters

Thanks to AMD’s SR-IOV tech, the MI300X GPU can be split into up to 8 virtual GPUs (VFs). That means:
- Multiple VMs or users can share a single physical GPU, securely and efficiently.
- Each VF gets its own slice of memory and compute—perfect for cloud, research, or enterprise AI.

In this project, the VM uses 2 VFs, which is plenty to run a 32B parameter LLM and still have room for other tasks.

---

## ⚡ Real-World Benefits

With a whopping 192GB of HBM3 memory and 304 compute units, the MI300X is built for heavy-duty LLM inference. This project shows you can:
- Run **interactive AI assistants** in secure, multi-tenant environments
- Do **batch processing** for huge prompt datasets
- **Monitor resources live** for better operations and tuning

---

## 🏁 Quick Start: Commands
Clone the repository
git clone https://github.com/dumroo/MI300X-Inference-DSQ-32B.git cd MI300X-Inference-DSQ-32B
Download the model (inside the script or manually)
Run in chat mode
python3 DSQ-32B_Inference.py –mode chat
Run in batch mode
python3 DSQ-32B_Inference.py –mode batch –input_file prompts.txt –output_file completions.txt


---

## 🎉 Wrapping Up

The MI300X-Inference-DSQ-32B project is a great template for running massive LLMs on AMD’s latest GPUs, all inside a virtualized environment. With a simple command-line interface, live GPU stats, and robust virtualization, it’s a powerful solution for researchers, developers, and enterprises looking to push the boundaries of generative AI.

**Give it a try, watch your GPU in action, and see what’s possible with scalable, virtualized AI inference!**

---

**References:**
- [MI300X-Inference-DSQ-32B on GitHub](https://github.com/dumroo/MI300X-Inference-DSQ-32B)
- [DeepSeek-R1-Distill-Qwen-32B Model](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)
- [AMD ROCm Documentation](https://rocmdocs.amd.com/en/latest/)

---