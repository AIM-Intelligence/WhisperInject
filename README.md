# WhisperInject

**When Good Sounds Go Adversarial: Jailbreaking Audio-Language Models with Benign Inputs**

[Bodam Kim](mailto:bk@aim-intelligence.com), [Hiskias Dingeto](mailto:hiskias@aim-intelligence.com), [Taeyoun Kwon](mailto:taeyounkwon@aim-intelligence.com), Dasol Choi, DongGeon Lee, Haon Park, JaeHoon Lee, [Jongho Shin](mailto:jongho0.shin@lge.com)

📄 **Paper**: [arXiv:2508.03365](https://arxiv.org/abs/2508.03365)

## Abstract

As large language models become increasingly integrated into daily life, audio has emerged as a key interface for human-AI interaction. However, this convenience also introduces new vulnerabilities, making audio a potential attack surface for adversaries. Our research introduces WhisperInject, a two-stage adversarial audio attack framework that can manipulate state-of-the-art audio language models to generate harmful content. Our method uses imperceptible perturbations in audio inputs that remain benign to human listeners. The first stage uses a novel reward-based optimization method, Reinforcement Learning with Projected Gradient Descent (RL-PGD), to guide the target model to circumvent its own safety protocols and generate harmful native responses. This native harmful response then serves as the target for Stage 2, Payload Injection, where we use Projected Gradient Descent (PGD) to optimize subtle perturbations that are embedded into benign audio carriers, such as weather queries or greeting messages. Validated under the rigorous StrongREJECT, LlamaGuard, as well as Human Evaluation safety evaluation framework, our experiments demonstrate a success rate exceeding 86% across Qwen2.5-Omni-3B, Qwen2.5-Omni-7B, and Phi-4-Multimodal. Our work demonstrates a new class of practical, audio-native threats, moving beyond theoretical exploits to reveal a feasible and covert method for manipulating AI behavior.

## Key Findings

- **>86% success rate** across Qwen2.5-Omni-3B, Qwen2.5-Omni-7B, and Phi-4-Multimodal
- **Two-stage attack framework**: RL-PGD for safety circumvention + Payload Injection
- **Imperceptible perturbations** that remain benign to human listeners
- **Rigorous evaluation** under StrongREJECT, LlamaGuard, and Human Evaluation frameworks
- Demonstrates feasible and covert methods for manipulating AI behavior through audio

## Repository Status

🚧 **Codebase coming soon** 🚧

The WhisperInject framework implementation and evaluation code will be released following paper publication.

## Citation

```bibtex
@article{kim2025whisperinject,
  title={When Good Sounds Go Adversarial: Jailbreaking Audio-Language Models with Benign Inputs},
  author={Kim, Bodam and Dingeto, Hiskias and Kwon, Taeyoun and Choi, Dasol and Lee, DongGeon and Park, Haon and Lee, JaeHoon and Shin, Jongho},
  journal={arXiv preprint arXiv:2508.03365},
  year={2025}
}
```
