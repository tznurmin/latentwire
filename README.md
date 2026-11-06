# LatentWire

LatentWire introduces a potentially novel memory interface for a frozen language
model. It writes encodings calculated from the frozen weights directly into the
model's existing recurrent memory. A small trained reader connects the resulting
internal signal to ordinary text generation.

The fixed example below uses Qwen3.5-9B to retrieve a stored colour that never
appears in its input prompt. The model's original weights remain unchanged.

**Input prompt, identical in every run**

> Which colour is stored in the memory slot? Reply with one colour word,
> or unknown if the slot is empty.

Before the model processes the final prompt token, LatentWire writes a numerical
encoding of the colour into one recurrent-memory row. The attached reader
processes the measured signal; the model then generates its reply normally.

| Value written into state | Reader | Generated reply |
| --- | --- | --- |
| red | on | red |
| blue | on | blue |
| red | off | unknown |
| no write | on | unknown |

Here, the memory slot is the selected recurrent-state row. No write leaves the
state produced by the preceding prompt tokens unchanged.

The 2 MiB reader was trained to recover eight colour values: red, blue, green, pink,
purple, black, white and orange. Each condition starts from a fresh model cache
with the same prompt and reader weights. Displayed replies omit surrounding
whitespace and the end token; the accompanying data retains the complete replies
and evaluation records.

At each of three training seeds, the normalized reader recovered all 32 pairs of
trained colours and training questions. Three of four new question phrasings recovered all
eight trained colours; the fourth returned `unknown` throughout. The four tested
untrained colours scored 0/32 per seed. Readers trained on unnormalized signals
returned `unknown` for every written colour under the same training procedure.
Scoring uses exact first-token matching, distinguishing `unknown` from ` unknown`;
all no-write replies display `unknown` after removing surrounding whitespace and
the end token.

## Method

Qwen3.5-9B uses Gated DeltaNet recurrent layers [1]. For colour token $i$, the
vocabulary output weights $W$ and the selected head's output-projection columns
$O_h$ define its encoding:

```math
c_i = \frac{W_{i,:}\,O_h}{\left\lVert W_{i,:}\,O_h \right\rVert_2}
```

On the loaded model, $W$ is `model.lm_head.weight`. $O_h$ selects columns 0–127 of
`model.model.language_model.layers[16].linear_attn.out_proj.weight`.
During codeword preparation, projection and initial normalization use 32-bit
floating point (FP32). Each row is then renormalized in 64-bit arithmetic and
rounded to FP32 for the supplied artifact. Inference writes these stored values
unchanged.

The writer replaces row 7 of head 0 at layer 16 with $c_i$ before the final prompt
token is processed. All indices are zero-based. Earlier prompt tokens have
already populated the cache; the recurrent update retains ordinary decay.

The reader takes the measured 128-dimensional head output $y$, before the head's
gated normalization and output projection. It normalizes $y$ and adds a learned
projection to the residual $x$ immediately before layer 27:

```math
\tilde{y} = \mathrm{float32}(y), \qquad
x' = x + \mathrm{bf16}\!\left(
R\,\frac{\tilde{y}}{\max\!\left(\lVert\tilde{y}\rVert_2,\,10^{-12}\right)}
\right)
```

$R$ is a $4096 \times 128$ matrix stored in FP32, with no bias. BF16 denotes
bfloat16, the model's 16-bit floating-point format.

Both hooks detach after this forward pass. The model generates its reply by greedy
decoding for up to 16 tokens or until an end token. The chat template contains
one user message with thinking disabled.

Only $R$ is trained, using the eight colours with four question phrasings and
no-write examples targeting ` unknown`. The supplied reader uses seed 101:
zero initialization, 4,096 single-example AdamW updates, learning rate 0.003,
zero weight decay and gradient-norm clipping at 1. The objective is cross-entropy
on the first answer token. The model's original parameters remain frozen.

## Run the example

This release provides the fixed inference example, one trained reader
(normalized input, seed 101), codewords and all 1,776 evaluation records from
six trained reader configurations. Training and full-evaluation code are not
included.

Use Qwen/Qwen3.5-9B revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
The checkpoint directory must contain regular copies of only the files declared
in [the checkpoint manifest](latentwire/_qwen_checkpoint_data.py).
Download and extract the [v1.0.0 inputs and results](https://github.com/tznurmin/latentwire/releases/download/v1.0.0/latentwire-qwen3.5-9b-artifacts-v1.0.0.zip).
Use its `inputs/` directory, containing `reader.npy`, `codewords.npy` and
`manifest.json`, as the artifact path:

```sh
nix develop --command python example.py --model /absolute/checkpoint --artifacts /absolute/inputs
```

The reference runner requires x86-64 Linux and one visible CUDA GPU with native
BF16 support and at least 22 GiB of free GPU memory. On a multi-GPU machine,
`CUDA_VISIBLE_DEVICES` selects the device. The command verifies the published
inputs and runs the four conditions above.

The runtime uses PyTorch 2.13.0, Transformers 5.17.0, Tokenizers 0.23.2 and
Safetensors 0.8.0 with eager/reference CUDA kernels, BF16 model weights, FP32
projections and TensorFloat-32 disabled. The provided flake and lock specify
the numerical environment and offline settings.

## References

Gated DeltaNet supplies the recurrent state mechanism [1]. The In-context
Autoencoder learns an encoder that compresses text into memory slots consumed
by a frozen model [2]. Wang and Reid derive Jacobian readouts and intervention
directions for recurrent state [3]. Prometheus Mind uses normalized vocabulary
output weights as memory values injected through attention key–value pairs [4].
δ-mem adds a delta-rule memory whose readout produces learned attention
corrections in a frozen backbone [5].

LatentWire's specific contribution is a paired write-and-read construction:
encodings calculated from frozen vocabulary and head-projection weights replace
an existing recurrent-memory row, and a trained linear reader projects the
measured head output into a later residual. Each fresh-cache run can supply a
different trained colour while both the original model weights and the trained
reader remain fixed.

1. Yang, S., Kautz, J., and Hatamizadeh, A. (2025).
   [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464v3).
   ICLR.
2. Ge, T., Hu, J., Wang, L., Wang, X., Chen, S.-Q., and Wei, F. (2024).
   [In-context Autoencoder for Context Compression in a Large Language Model](https://arxiv.org/abs/2307.06945v4).
   ICLR.
3. Wang, W., and Reid, F. (2026).
   [Residual Streams Read, Recurrent States Remember: The Global Workspace in Mamba Models](https://arxiv.org/abs/2609.32102v1).
   arXiv.
4. Wind, M. (2026).
   [Prometheus Mind: Retrofitting Memory to Frozen Language Models](https://arxiv.org/abs/2601.15324v1).
   arXiv.
5. Lei, J., et al. (2026).
   [δ-mem: Efficient Online Memory for Large Language Models](https://arxiv.org/abs/2605.12357v2).
   arXiv.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
See [NOTICE](NOTICE) for copyright information.
