# 🎬 Multimodal Video Generation Models with Audio: Present and Future

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org)
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Abstract:** Video generation models have advanced rapidly and are now widely used across entertainment, advertising, filmmaking, and robotics applications such as world modeling and simulation. However, visual content alone is often insufficient for realistic and engaging media experiences—audio is a key component of immersion and semantic coherence. As AI-generated videos become increasingly prevalent in everyday content, demand has grown for systems that can generate synchronized sound alongside visuals. This trend has driven rising interest in **multimodal video generation**, which jointly models video and audio to produce more complete, coherent, and appealing outputs.

---

## 📖 Table of Contents

- [📚 Citation](#-citation)
- [✨ Representative Models](#-representative-models)
- [🏗️ Architectures & Evolution](#-architectures--evolution)
  - [1. Variational Autoencoder (VAE)](#1-variational-autoencoder-vae)
  - [2. U-Net Architectures](#2-u-net-architectures)
  - [3. Diffusion Transformer (DiT)](#3-diffusion-transformer-dit)
  - [4. Future: Mixture of Experts (MoE)](#4-future-mixture-of-experts-moe)
- [⚙️ Post-Training & Alignment](#-post-training--alignment)
- [📊 Evaluation](#-evaluation)
- [🚀 Applications & Research Directions](#-applications--research-directions)
  - [Video-to-Audio (V2A)](#video-to-audio-v2a)
  - [Streaming Generation](#streaming-generation)
  - [Human-Centric Generation](#human-centric-generation)
  - [Long Video Generation](#long-video-generation)
  - [World Models](#world-models)

---

## 📚 Citation

If you find this survey useful for your research, please cite our work:

```bibtex
@article{author2025multimodal,
  title={Multimodal Video Generation Models with Audio: Present and Future},
  author={Author, First A. and Author, Second B. and Author, Third C.},
  journal={IEEE Access},
  year={2025},
  doi={10.1109/ACCESS.2017.DOI}
}
```

---

## ✨ Representative Models

A summary of current state-of-the-art multimodal video diffusion models discussed in our survey.

| Model / System                                                            | Architecture             | Key Features                                   | Release   |
| :------------------------------------------------------------------------ | :----------------------- | :--------------------------------------------- | :-------- |
| **[Google Veo 3.1](https://deepmind.google/models/veo/)**                 | Diffusion + Sync Audio   | Native audiovisual synthesis                   | May 2025  |
| **[OpenAI Sora 2](https://openai.com/index/sora-2/)**                     | DiT (Enhanced)           | Improved temporal coherence; native audio sync | Sep 2025  |
| **Grok 4**                                                                | -                        | -                                              | July 2025 |
| **[Wan 2.6](https://www.xrmm.com/)**                                      | MoE DiT + MM Transformer | Simultaneous audio-visual generation           | Dec 2025  |
| **[Kling 2.6](https://klingai.com/global/)**                              | DiT + MM Transformer     | Simultaneous audio-visual generation           | Dec 2025  |
| **[MM-Diffusion](https://github.com/researchmm/MM-Diffusion)** $^\dagger$ | Decoupled U-Net          | Early joint generation foundation model        | 2023      |
| **[OVI](https://huggingface.co/spaces/akhaliq/Ovi)** $^\dagger$           | DiT + Sync Audio-Video   | Native 4K @ 50fps; open-source foundation      | Oct 2025  |
| **[LTX-2](https://huggingface.co/Lightricks/LTX-2)** $^\dagger$           | DiT + Sync Audio-Video   | Native 4K @ 50fps; open-source foundation      | Jan 2026  |

> $^\dagger$ _Denotes Open-Source Models_

---

## 🏗️ Architectures & Evolution

Multimodal video generation requires synchronizing distinct modalities (visual frames and audio waveforms) within a unified architecture. We trace the evolution from foundational VAEs to modern DiT and MoE architectures.

### 1. Variational Autoencoder (VAE)

**Overview:** VAEs establish a probabilistic mapping between input data and a latent space. In modern multimodal systems, they primarily serve as **compression mechanisms** (Video VAE & Audio VAE) to transform high-dimensional raw data into compact latent representations for efficient processing.

| Component             | Function                                 | Key Characteristics                             |
| :-------------------- | :--------------------------------------- | :---------------------------------------------- |
| **Video VAE Encoder** | Compresses raw frames into video latents | 3D Encoder with spatial & temporal convolutions |
| **Audio VAE Encoder** | Transforms waveforms into audio latents  | Encodes acoustic features & temporal dynamics   |

<div align="center">
  <img src="fig/VAE.png" width="40%" alt="VAE Architecture"/>
  <br>
  <em>Figure 1: Variational Autoencoder (VAE) architecture used for latent space encoding.</em>
</div>

### 2. U-Net Architectures

**Overview:** Originally designed for segmentation, the U-Net's encoder-decoder structure with skip connections became the backbone for early diffusion models. For multimodal generation, **Coupled U-Nets** are used to jointly denoise video and audio streams.

| Model                                                          | Architecture  | Key Features                                                          | Paper                                         |
| :------------------------------------------------------------- | :------------ | :-------------------------------------------------------------------- | :-------------------------------------------- |
| **[MM-Diffusion](https://github.com/researchmm/MM-Diffusion)** | Coupled U-Net | Two parallel U-Net subnets (Video & Audio) with cross-modal attention | [CVPR 2023](https://arxiv.org/abs/2305.14524) |
| **MM-LDM**                                                     | Latent U-Net  | Operates in shared latent space to reduce computational costs         | [Paper](https://arxiv.org)                    |

<div align="center">
  <img src="fig/Unet.drawio.svg" width="40%" alt="U-Net Architecture"/>
  <br>
  <em>Figure 2: U-Net architecture, the foundation for early joint generation models like MM-Diffusion.</em>
</div>

### 3. Diffusion Transformer (DiT)

**Overview:** The current industry standard. DiT replaces U-Net with Transformer blocks, enabling better scalability and global spatiotemporal reasoning. Modern systems use **Dual-Stream Fusion**, where video and audio streams communicate bidirectionally via cross-attention (A2V and V2A).

| Model                                                | Architecture     | Key Features                                   | Paper                      |
| :--------------------------------------------------- | :--------------- | :--------------------------------------------- | :------------------------- |
| **[OVI](https://huggingface.co/spaces/akhaliq/Ovi)** | DiT + Sync Audio | Native 4K @ 50fps; Open-source foundation      | [Paper](https://arxiv.org) |
| **[LTX-2](https://huggingface.co/Lightricks/LTX-2)** | DiT + Sync Audio | Native 4K @ 50fps; Open-source foundation      | [Paper](https://arxiv.org) |
| **[Sora 2](https://openai.com/index/sora-2/)**       | Enhanced DiT     | Improved temporal coherence; native audio sync | -                          |
| **[Veo 3.1](https://deepmind.google/models/veo/)**   | Diffusion + Sync | Native audiovisual synthesis                   | -                          |

<div align="center">
  <img src="fig/DiT.svg" width="40%" alt="DiT Architecture"/>
  <img src="fig/architecture.png" width="90%" alt="Architecture Evolution"/>
  <br>
  <em>Figure 3: Diffusion Transformer (DiT) and the evolution towards native audio-visual synthesis with Dual-Stream Fusion.</em>
</div>

### 4. Future: Mixture of Experts (MoE)

**Overview:** To scale to billions of parameters efficiently, **Mixture of Experts (MoE)** architectures introduce sparse activation. Only a subset of parameters (experts) is activated for each input token, allowing models to handle high-noise (global structure) and low-noise (fine details) phases with specialized experts.

| Model                                | Architecture | Key Features                                                        | Paper                      |
| :----------------------------------- | :----------- | :------------------------------------------------------------------ | :------------------------- |
| **[Wan 2.6](https://www.xrmm.com/)** | MoE DiT      | Sparse activation for efficient scaling; simultaneous AV generation | [Paper](https://arxiv.org) |

---

## ⚙️ Post-Training & Alignment

Pre-trained base models often require adaptation for precise audio-visual synchronization. Key techniques include:

### 1. Parameter-Efficient Fine-Tuning (PEFT)

- **LoRA (Low-Rank Adaptation):** Injects lightweight trainable matrices into attention layers (e.g., **AV-DiT**).
- **Adapters:** Semantic adapters for conditioning audio on video features; Temporal adapters for precise timestamp control (e.g., **FoleyCrafter**).

### 2. Audio-Visual Alignment Modules

- **Synchformer:** Self-supervised audio-visual desynchronization detector for millisecond-level alignment.
- **ControlNet-Based Methods:**
  - **Temporal ControlNet:** Enforces alignment using timestamp masks.
  - **Multi-Stream Control:** Separates speech, music, and effects for fine-grained control (e.g., **MTV**).

### 3. Training-Free Methods

- **Audio-Visual Guidance:** Manipulates attention scores during inference to steer video generation towards audio synchronization without weight updates.

<div align="center">
  <img src="fig/Post-Training-Methods.svg" width="90%" alt="Post Training Methods"/>
  <br>
  <em>Figure 4: Common post-training methods including PEFT, Alignment Modules, and Attention Injection.</em>
</div>

---

## 📊 Evaluation

Evaluating joint video-audio generation is complex, requiring assessments of video quality, audio quality, and cross-modal alignment.

| Paradigm         | Category          | Metrics / Aspects                                                                        |
| :--------------- | :---------------- | :--------------------------------------------------------------------------------------- |
| **Quantitative** | **Video Quality** | **FVD**, **CLIPScore**, **VBench-2.0** (physics, human fidelity), **VBench++**           |
|                  | **Audio Quality** | **FAD**, **KL Divergence**, **PAM** (Prompt Adherence), **Audiobox Aesthetics**, **MAD** |
|                  | **AV Alignment**  | **AV-Align**, **DeSync** (temporal misalignment), **ImageBind Score**, **FAVD**          |
| **Qualitative**  | **Protocol**      | **MOS** (Mean Opinion Score), Side-by-Side Preference                                    |
|                  | **Aspects**       | Temporal Coherence, Sound Relevance, Spatial Consistency                                 |

<div align="center">
  <img src="fig/Video-Audio GenerationEvaluation.svg" width="90%" alt="Evaluation Framework"/>
  <br>
  <em>Figure 5: Multimodal Evaluation Common Practices.</em>
</div>

---

## 🚀 Applications & Research Directions

The field is expanding into diverse domains, moving beyond silent video to immersive audiovisual experiences.

<div align="center">
  <img src="fig/application.svg" width="100%" alt="Applications"/>
  <br>
  <em>Figure 6: Mainstream Multimodal Video Generation Research Areas.</em>
</div>

### Video-to-Audio (V2A)

Synthesizing synchronized sound for existing silent videos.

- **Models:** **Diff-Foley** (Contrastive Pretraining), **MMAudio** (Joint Training).

### Streaming Generation

Real-time, low-latency generation for live interactions.

- **Techniques:** Causal temporal modeling, Sliding-window attention.
- **Models:** **StreamDiffusion**, **CausVid**, **MotionStream**.

### Human-Centric Generation

Generating realistic human avatars with synchronized speech and gestures.

- **Face Animation:** **OmniHuman-1** (Full-body), **EMO**, **VLOGGER**.
- **Pose Animation:** **Wan Animate**, **Animate Anyone**.

### Long Video Generation

Maintaining coherence over minutes or unbounded lengths.

- **Techniques:** Rolling Forcing, Context Caching.
- **Models:** **StreamingT2V**, **FramePack**, **LongLive**.

### World Models

Simulating physics and acoustics for embodied AI.

- **Generative World Models:** **Movie Gen** (Meta), **Veo 3** (Google), **AV-CDiT** (Embodied AI simulation).
- **Goal:** Unified simulation of visual dynamics and acoustic environments (spatial sound, reverberation).
