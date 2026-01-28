# 👁️ DeepScene: AI Image Narrator

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/Transformers-BLIP--2-yellow?style=for-the-badge&logo=huggingface)
![BitsAndBytes](https://img.shields.io/badge/Optimization-8--bit-success?style=for-the-badge)

A **Deep Learning Image Analysis Tool** capable of generating human-level narrative descriptions of static images.

This project utilizes the **BLIP-2 (Bootstrapping Language-Image Pre-training)** model architecture. It is specifically engineered to run the heavy **OPT-2.7b** model on consumer laptops (like the RTX 4050/3050) by implementing **8-bit quantization** and intelligent **CPU offloading**.

---

## 🚀 Key Features

* **🧠 Deep Scene Understanding:** Goes beyond simple captions. It generates detailed, multi-sentence paragraphs describing the subject, background, lighting, and atmosphere.
* **📉 Resource Optimization:** Uses `bitsandbytes` to load a 15GB+ model into ~6GB of VRAM using 8-bit quantization, making high-end AI accessible on mid-range hardware.
* **🖥️ User-Friendly GUI:** Includes a built-in **Tkinter File Picker**—no need to type long file paths in the terminal.
* **🛡️ Anti-Repetition Logic:** Custom generation parameters prevent the model from getting stuck in repetitive loops (a common issue with OPT models).
* **🔀 Dual Modes:**
    * **Paragraph Mode:** For storytelling and detailed analysis.
    * **Phrase Mode:** For quick, short distinct captions.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10
* **Core Model:** `Salesforce/blip2-opt-2.7b`
* **Frameworks:**
    * `torch` (PyTorch)
    * `transformers` (Hugging Face)
    * `accelerate` (Memory Management)
    * `bitsandbytes` (Int8 Quantization)
    * `tkinter` (Native GUI)

---

## 💻 Installation

### Prerequisites
* NVIDIA GPU (Recommended for speed, but runs on CPU)
* Python 3.10

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/DeepScene-AI.git](https://github.com/your-username/DeepScene-AI.git)
cd DeepScene-AI

```
## 🚀 How to Run

You can run the program in two ways:

### Method 1: Interactive Mode (Easiest)
Simply run the script without any arguments. A window will pop up to help you.

```bash
python first.py

### Method 2: Command Line (Fastest)
# Syntax: python first.py "path/to/image.jpg" [mode]
python first.py "C:\Images\mountain.jpg" p
