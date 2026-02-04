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

Multimodal video generation requires synchronizing distinct modalities (visual frames and audio waveforms) within a unified architecture. We trace the evolution from foundational VAEs to modern DiT and MoE architectures. Below is a comprehensive categorization of the models and papers referenced in our survey.

### 1. Variational Autoencoder (VAE)

VAEs establish a probabilistic mapping between input data and a latent space. In modern multimodal systems, they primarily serve as **compression mechanisms** (Video VAE & Audio VAE) to transform high-dimensional raw data into compact latent representations.

| Paper / Model                           | Category            | Author                 | Year | Key Contribution                                        |
| :-------------------------------------- | :------------------ | :--------------------- | :--- | :------------------------------------------------------ |
| **Auto-Encoding Variational Bayes**     | **Foundational**    | Kingma & Welling       | 2013 | Introduces VAEs: probabilistic mapping to latent space. |
| **Reducing the dimensionality of data** | **Foundational**    | Hinton & Salakhutdinov | 2006 | Traditional autoencoders mapping to fixed vectors.      |
| **MM-Diffusion (VAE Usage)**            | **Video/Audio VAE** | Ruan et al.            | 2023 | Uses separate VAE encoders for video and audio streams. |

<div align="center">
  <img src="fig/VAE.png" width="90%" alt="VAE Architecture"/>
  <br>
  <em>Figure 1: Variational Autoencoder (VAE) architecture used for latent space encoding.</em>
</div>

### 2. Diffusion Architecture (U-Net & Early Models)

Diffusion models reverse a gradual noising process. Early architectures relied on **U-Nets** with skip connections, which were effective but limited in scalability for long-sequence multimodal tasks.

| Paper / Model                                       | Category             | Author             | Year | Key Contribution                                                  |
| :-------------------------------------------------- | :------------------- | :----------------- | :--- | :---------------------------------------------------------------- |
| **Denoising Diffusion Probabilistic Models (DDPM)** | **Foundational**     | Ho et al.          | 2020 | Core framework for training generative models via denoising.      |
| **Diffusion Models Beat GANs**                      | **Foundational**     | Dhariwal & Nichol  | 2021 | Demonstrates diffusion superiority in image synthesis.            |
| **Video Diffusion Models**                          | **Foundational**     | Ho et al.          | 2022 | Extends diffusion to video domain.                                |
| **Imagen Video**                                    | **Foundational**     | Ho et al.          | 2022 | High-definition video generation using cascaded diffusion models. |
| **Denoising Diffusion Implicit Models (DDIM)**      | **Foundational**     | Song et al.        | 2022 | Accelerates sampling for diffusion models.                        |
| **Latent Diffusion Models (LDM)**                   | **Foundational**     | Rombach et al.     | 2022 | Performs diffusion in compressed latent space for efficiency.     |
| **U-Net**                                           | **U-Net Backbone**   | Ronneberger et al. | 2015 | Original encoder-decoder architecture with skip connections.      |
| **MM-Diffusion**                                    | **Coupled U-Net**    | Ruan et al.        | 2023 | Coupled U-Nets for joint video-audio denoising.                   |
| **MM-LDM**                                          | **Latent U-Net**     | Sun et al.         | 2024 | Multi-modal latent diffusion in shared semantic space.            |
| **Diff-Foley**                                      | **Audio-Visual**     | Luo et al.         | 2023 | Synchronized video-to-audio synthesis with LDMs.                  |
| **AudioLDM**                                        | **Audio-Visual**     | Liu et al.         | 2023 | Text-to-audio generation using LDMs and CLAP.                     |
| **Align Your Latents**                              | **Latent Alignment** | Blattmann et al.   | 2023 | High-resolution video synthesis with latent diffusion.            |
| **MSF**                                             | **Optimization**     | Xu et al.          | 2025 | Efficient diffusion via multi-scale latent factorization.         |

<div align="center">
  <img src="fig/Unet.drawio.svg" width="90%" alt="U-Net Architecture"/>
  <br>
  <em>Figure 2: U-Net architecture, the foundation for early joint generation models like MM-Diffusion.</em>
</div>

### 3. Diffusion Transformer (DiT)

The current industry standard. DiT replaces U-Net with **Transformer** blocks, enabling better scalability and global spatiotemporal reasoning. Modern systems use **Dual-Stream Fusion** for bidirectional audio-video communication.

| Paper / Model                                   | Category              | Author         | Year | Key Contribution                                               |
| :---------------------------------------------- | :-------------------- | :------------- | :--- | :------------------------------------------------------------- |
| **Scalable Diffusion Models with Transformers** | **Core Architecture** | Peebles & Xie  | 2023 | Introduces DiT: Transformers as diffusion backbones.           |
| **DiT (ICCV)**                                  | **Core Architecture** | Peebles & Xie  | 2023 | Scalable Diffusion Models with Transformers (Conference).      |
| **Attention Is All You Need**                   | **Core Architecture** | Vaswani et al. | 2017 | Foundational Transformer architecture with self-attention.     |
| **Latte**                                       | **Video DiT**         | Ma et al.      | 2024 | Latent Diffusion Transformer specialized for video generation. |
| **OVI**                                         | **Multimodal DiT**    | Low et al.     | 2025 | Twin backbone cross-modal fusion for native AV generation.     |
| **LTX-2**                                       | **Multimodal DiT**    | HaCohen et al. | 2026 | Efficient joint audio-visual foundation model (Native 4K).     |
| **MMAudio**                                     | **Multimodal DiT**    | Cheng et al.   | 2025 | Taming multimodal joint training for high-quality synthesis.   |
| **T5**                                          | **Conditioning**      | Raffel et al.  | 2020 | Text-to-text transfer transformer used for text encoding.      |
| **CLIP**                                        | **Conditioning**      | Radford et al. | 2021 | Contrastive language-image pretraining for semantic alignment. |
| **RoFormer (RoPE)**                             | **Positional Enc.**   | Su et al.      | 2024 | Rotary Position Embedding for temporal alignment.              |

<div align="center">
  <img src="fig/DiT.svg" width="90%" alt="DiT Architecture"/>
  <img src="fig/architecture.png" width="90%" alt="Architecture Evolution"/>
  <br>
  <em>Figure 3: Diffusion Transformer (DiT) and the evolution towards native audio-visual synthesis with Dual-Stream Fusion.</em>
</div>

### 4. Future: Mixture of Experts (MoE)

To scale to billions of parameters efficiently, **Mixture of Experts (MoE)** architectures introduce sparse activation. Only a subset of parameters (experts) is activated for each input token.

| Paper / Model                          | Category             | Author          | Year | Key Contribution                                       |
| :------------------------------------- | :------------------- | :-------------- | :--- | :----------------------------------------------------- |
| **Wan 2.6**                            | **Video MoE**        | Team Wan et al. | 2025 | Large-scale MoE DiT for video generation.              |
| **Cosmos World**                       | **World Model**      | NVIDIA et al.   | 2025 | Foundation model platform for physical AI with MoE.    |
| **Outrageously Large Neural Networks** | **Foundational MoE** | Shazeer et al.  | 2017 | Introduces sparsely-gated Mixture-of-Experts layer.    |
| **GShard**                             | **Foundational MoE** | Lepikhin et al. | 2020 | Scaling giant models with conditional computation.     |
| **Switch Transformers**                | **Foundational MoE** | Fedus et al.    | 2022 | Scaling to trillion parameters with simple sparsity.   |
| **Scaling Vision with Sparse MoE**     | **Vision MoE**       | Riquelme et al. | 2021 | Applying MoE to vision tasks.                          |
| **Uni-MoE**                            | **Multimodal MoE**   | Li et al.       | 2025 | Scaling unified multimodal LLMs with MoE.              |
| **DeepSeekMoE**                        | **LLM MoE**          | Dai et al.      | 2024 | Ultimate expert specialization in MoE language models. |

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
