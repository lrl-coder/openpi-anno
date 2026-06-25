# pi0 模型输入到输出、VLM/Action Expert 信息流与 Loss 代码追踪

本文基于论文 `openpi/paper/Black_等_-_2026_-_π_0_A_Vision-Language-Action_Flow_Model_for_General_Robot_Control.md` 与当前仓库代码，追踪 `pi0` 从数据输入、VLM 与 action expert 的信息交互，到训练 loss 与推理输出 action chunk 的完整流程。

说明：本文以 JAX/Flax NNX 实现 `openpi/src/openpi/models/pi0.py` 为主线。仓库中还有 PyTorch 复刻 `openpi/src/openpi/models_pytorch/pi0_pytorch.py`，其关键逻辑与 JAX 版一致，本文在关键节点附带对照。

## 1. 论文定义与代码对象的对应关系

论文中，pi0 要建模：

$$
\begin{aligned}
p(A_t \mid o_t), \qquad
o_t &= [I_t^1,\ldots,I_t^n,\ell_t,q_t], \\
A_t &= [a_t,a_{t+1},\ldots,a_{t+H-1}].
\end{aligned}
$$

论文依据：

- 论文正文指出 observation 由多张 RGB 图像、语言命令、机器人本体状态组成，action 是未来 $H$ 步 action chunk；见论文 `openpi/paper/Black_等_-_2026_-_π_0_A_Vision-Language-Action_Flow_Model_for_General_Robot_Control.md:75`。
- 论文附录说明 PaliGemma 原本接收图像序列和语言 prompt，pi0 增加了 proprioceptive state $q_t$ 与 noisy action chunk $A_t^\tau$，并只使用 noisy action 对应的 transformer 输出投影成 vector field；见论文 `openpi/paper/Black_等_-_2026_-_π_0_A_Vision-Language-Action_Flow_Model_for_General_Robot_Control.md:411-415`。
- 论文附录说明 blockwise causal attention 有 3 个块：`[images, language]`、`[state]`、`[noisy actions]`；见论文 `openpi/paper/Black_等_-_2026_-_π_0_A_Vision-Language-Action_Flow_Model_for_General_Robot_Control.md:417`。
- 论文附录说明图像和语言路由到较大的 VLM backbone，状态和动作路由到 action expert；见论文 `openpi/paper/Black_等_-_2026_-_π_0_A_Vision-Language-Action_Flow_Model_for_General_Robot_Control.md:419`。

代码中的对应对象：

- `Observation` 数据结构在 `openpi/src/openpi/models/model.py:81-107` 定义，包含 `images`、`image_masks`、`state`、`tokenized_prompt`、`tokenized_prompt_mask`。
- `Actions` 类型在 `openpi/src/openpi/models/model.py:139-141` 定义，形状是 $[*b, ah, ad]$，即 batch 维、action horizon、action dim。
- pi0 默认 `action_dim=32`、`action_horizon=50`，见 `openpi/src/openpi/models/pi0_config.py:24-27`；输入 spec 明确三路图像、state、tokenized prompt 和 action chunk，见 `openpi/src/openpi/models/pi0_config.py:64-85`。
- 模型固定期望三路图像 key：`base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb`，分辨率 `224x224`，见 `openpi/src/openpi/models/model.py:38-47`。

## 2. 数据如何变成模型输入

### 2.1 原始样本到统一字段

训练数据首先由 dataset loader 取出。对 LeRobot 数据，`create_torch_dataset` 会用 `delta_timestamps` 按 `action_horizon` 为每个当前时刻构造未来动作序列：

- `openpi/src/openpi/training/data_loader.py:140-146`：`delta_timestamps={key: [t / fps for t in range(action_horizon)]}`，这一步把单步动作字段扩展成长度为 $H$ 的 action chunk。
- `openpi/src/openpi/training/data_loader.py:172-191`：训练数据随后应用 `repack_transforms`、`data_transforms`、`Normalize`、`model_transforms`。
- RLDS/DROID 路径类似，`create_rlds_dataset` 把 `action_chunk_size=action_horizon` 传入 `DroidRldsDataset`，见 `openpi/src/openpi/training/data_loader.py:154-169`。

不同 embodiment 的原始字段先被 repack 成统一字段。例如：

- ALOHA：`openpi/src/openpi/training/config.py:240-255` 将 `observation.images.top`、`observation.state`、`action` 映射到统一的 `images/state/actions`。
- LIBERO：`openpi/src/openpi/training/config.py:301-310` 将 `observation/image`、`observation/wrist_image`、`observation/state`、`actions`、`prompt` 重排。
- DROID RLDS：`openpi/src/openpi/training/config.py:383-393` 将外部相机、腕部相机、关节、夹爪、动作和 prompt 重排。

这些 repack 的执行函数是 `RepackTransform.__call__`，它 flatten 原字典并按目标结构取值，见 `openpi/src/openpi/transforms.py:80-101`。

### 2.2 embodiment-specific transform 到 pi0 标准三图像输入

以 LIBERO 为例：

- `openpi/src/openpi/policies/libero_policy.py:52-63`：从 `observation/image` 和 `observation/wrist_image` 解析出 base 与 wrist 图像，并用零图像补齐不存在的 `right_wrist_0_rgb`。
- `openpi/src/openpi/policies/libero_policy.py:64-69`：给每路图像附上 `image_mask`；对 pi0，补齐的右腕图像 mask 为 `False`。
- `openpi/src/openpi/policies/libero_policy.py:72-81`：训练时保留 `actions`，同时把语言指令放到 `prompt`。

以 DROID 为例：

- `openpi/src/openpi/policies/droid_policy.py:36-40`：把 `joint_position` 和 `gripper_position` 拼成低维 `state`。
- `openpi/src/openpi/policies/droid_policy.py:47-63`：对 `PI0 | PI05`，输出三路图像 key：base、left wrist、right wrist，其中不存在的 right wrist 用零图补齐且 mask 为 `False`。
- `openpi/src/openpi/policies/droid_policy.py:66-72`：训练时保留 `actions`，并传递 prompt。

### 2.3 action/state 归一化、delta action 与 padding

进入模型前，数据会被归一化：

- norm stats 的加载在 `openpi/src/openpi/training/config.py:179-199`。
- transform pipeline 中 normalization 位于 model transform 之前，见 `openpi/src/openpi/training/data_loader.py:183-190`。
- `Normalize` 对普通 z-score 使用 $(x-\mathrm{mean})/(\mathrm{std}+10^{-6})$，见 `openpi/src/openpi/transforms.py:114-140`；quantile 归一化见 `openpi/src/openpi/transforms.py:141-145`。

如果数据源是绝对关节目标，pi0 通常把动作转换为相对当前 state 的 delta action：

- `DeltaActions` 在 `openpi/src/openpi/transforms.py:203-221` 中实现：`actions[..., :dims] -= state[..., :dims]`，只对 mask 指定维度生效。
- ALOHA 默认启用 delta joint action，见 `openpi/src/openpi/training/config.py:229-268`。
- DROID 如果使用 `JOINT_POSITION` action space，也会转换为 delta action，见 `openpi/src/openpi/training/config.py:403-409`。

最后，`PadStatesAndActions` 会把 state 和 actions 的最后一维补齐到模型的 `action_dim`：

- `openpi/src/openpi/transforms.py:327-337`：`state` 必补，`actions` 存在时也补。
- 这解释了为什么 LIBERO 输出只取前 7 维，见 `openpi/src/openpi/policies/libero_policy.py:95-100`；DROID 输出只取前 8 维，见 `openpi/src/openpi/policies/droid_policy.py:77-81`。

### 2.4 prompt tokenization

pi0 的 model transform 在 `ModelTransformFactory` 中定义：

- `openpi/src/openpi/training/config.py:113-125`：对 `ModelType.PI0`，依次执行 `InjectDefaultPrompt`、`ResizeImages(224,224)`、`TokenizePrompt(PaligemmaTokenizer)`、`PadStatesAndActions`。
- `TokenizePrompt.__call__` 在 `openpi/src/openpi/transforms.py:247-266`：取出 `prompt`，调用 tokenizer，产出 `tokenized_prompt` 与 `tokenized_prompt_mask`。
- `PaligemmaTokenizer.tokenize` 在 `openpi/src/openpi/models/tokenizer.py:22-48`：pi0 模式下不把 state 放进文本；它对清洗后的 prompt 加 BOS，再额外编码换行 token 作为 start-of-answer 标记，随后 padding/truncation 到 `max_token_len`。

到这里，一个训练 batch 被整理成：

- `Observation.images["base_0_rgb"]`: $[B,224,224,3]$
- `Observation.images["left_wrist_0_rgb"]`: $[B,224,224,3]$
- `Observation.images["right_wrist_0_rgb"]`: $[B,224,224,3]$
- `Observation.image_masks[...]`: 每路图像 $[B]$
- `Observation.state`: $[B,\mathrm{action\_dim}]$
- `Observation.tokenized_prompt`: $[B,\mathrm{max\_token\_len}]$
- `Observation.tokenized_prompt_mask`: $[B,\mathrm{max\_token\_len}]$
- `Actions.actions`: $[B,\mathrm{action\_horizon},\mathrm{action\_dim}]$

`Observation.from_dict` 会把 nested dict 转成结构化对象，同时把 uint8 图像转换到 `[-1, 1]` float，见 `openpi/src/openpi/models/model.py:109-129`。

## 3. 模型构造：PaliGemma VLM + action expert

### 3.1 配置

`Pi0Config` 的默认关键参数：

- `paligemma_variant="gemma_2b"`，`action_expert_variant="gemma_300m"`，见 `openpi/src/openpi/models/pi0_config.py:20-23`。
- `action_dim=32`、`action_horizon=50`，见 `openpi/src/openpi/models/pi0_config.py:24-27`。
- `max_token_len` 默认 `48`，见 `openpi/src/openpi/models/pi0_config.py:37-41`。

Gemma 配置：

- `gemma_2b` 在代码中宽度 `2048`、深度 `18`、MLP `16384`，见 `openpi/src/openpi/models/gemma.py:79-87`。
- `gemma_300m` 在代码中宽度 `1024`、深度 `18`、MLP `4096`，见 `openpi/src/openpi/models/gemma.py:69-78`。
- 注意：论文附录 `openpi/paper/Black_等_-_2026_-_π_0_A_Vision-Language-Action_Flow_Model_for_General_Robot_Control.md:419` 的文字列出了 PaliGemma/Gemma 配置；当前代码是最终依据，尤其 `num_heads` 在 `gemma.py` 中为 8。

### 3.2 模型初始化

`Pi0.__init__` 中完成三类模块：

- `openpi/src/openpi/models/pi0.py:70-80`：读取 PaliGemma 和 action expert 的 Gemma config，创建 `_gemma.Module(configs=[paligemma_config, action_expert_config])`。这就是一个 transformer 内部的两套 expert 权重。
- `openpi/src/openpi/models/pi0.py:81-91`：创建 SigLIP 图像 encoder，并把 `llm` 与 `img` 放入 `self.PaliGemma`。
- `openpi/src/openpi/models/pi0.py:92-100`：创建机器人侧投影层。pi0 主线包括：
  - `action_in_proj`: $\mathrm{action\_dim}\rightarrow\mathrm{action\_expert\_width}$
  - `state_proj`: $\mathrm{action\_dim}\rightarrow\mathrm{action\_expert\_width}$
  - `action_time_mlp_in`: $2\cdot\mathrm{action\_expert\_width}\rightarrow\mathrm{action\_expert\_width}$
  - `action_time_mlp_out`: $\mathrm{action\_expert\_width}\rightarrow\mathrm{action\_expert\_width}$
  - `action_out_proj`: $\mathrm{action\_expert\_width}\rightarrow\mathrm{action\_dim}$

PyTorch 对照：`openpi/src/openpi/models_pytorch/pi0_pytorch.py:90-109` 同样构建 PaliGemma expert、action projection、state projection 和 time/action MLP。

## 4. Prefix：图像与语言如何进入 VLM backbone

`Pi0.embed_prefix` 负责论文中的 $[I_t^1,\ldots,I_t^n,\ell_t]$：

1. 遍历 `obs.images`，每路图像进入 SigLIP：
   - `openpi/src/openpi/models/pi0.py:112-116`：`image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)`。
   - 图像 mask 从 $[B]$ 扩展为 $[B,\mathrm{image\_token\_count}]$，见 `openpi/src/openpi/models/pi0.py:117-123`。
   - `ar_mask += [False] * image_tokens.shape[1]`，见 `openpi/src/openpi/models/pi0.py:124-125`，表示图像 token 之间处于同一个 attention block，可双向互看。

2. 如果有语言 token，则进入 PaliGemma/Gemma embedding table：
   - `openpi/src/openpi/models/pi0.py:127-131`：`self.PaliGemma.llm(obs.tokenized_prompt, method="embed")` 得到 language token embedding，并附上 prompt mask。
   - `openpi/src/openpi/models/pi0.py:132-133`：语言 token 的 `ar_mask` 也追加 `False`，因此图像与语言属于同一个 prefix block。

3. prefix tokens、mask、ar mask 拼接返回：
   - `openpi/src/openpi/models/pi0.py:134-137`。

从这一步可以推出：图像和语言都作为 `prefix_tokens` 进入 expert 0，也就是 PaliGemma/VLM backbone。调用点见训练 forward 的 `self.PaliGemma.llm([prefix_tokens, suffix_tokens], ...)`，其中 list 的第 0 个元素对应 expert 0；见 `openpi/src/openpi/models/pi0.py:209-211`。

PyTorch 对照：`openpi/src/openpi/models_pytorch/pi0_pytorch.py:187-236`。

## 5. Suffix：state、noisy actions 与 flow timestep 如何进入 action expert

`Pi0.embed_suffix` 负责论文中的 $[q_t,A_t^\tau]$：

1. pi0 主线先添加 state token：
   - `openpi/src/openpi/models/pi0.py:151-155`：`state_token = self.state_proj(obs.state)[:, None, :]`，把 $[B,\mathrm{action\_dim}]$ 投影成一个 $[B,1,\mathrm{action\_expert\_width}]$ token。
   - `openpi/src/openpi/models/pi0.py:156-157`：追加 `ar_mask=[True]`。根据 `make_attn_mask` 的 cumulative block 逻辑，这会开启一个新 block，因此 prefix 不能反向 attend 到 state。

2. noisy action chunk 先逐步投影：
   - `openpi/src/openpi/models/pi0.py:159`：`action_tokens = self.action_in_proj(noisy_actions)`，形状从 $[B,H,\mathrm{action\_dim}]$ 变成 $[B,H,\mathrm{action\_expert\_width}]$。

3. flow timestep 经过正弦位置编码：
   - `openpi/src/openpi/models/pi0.py:160-161`：`posemb_sincos(timestep, width, min_period=4e-3, max_period=4.0)`。
   - `posemb_sincos` 的具体实现见 `openpi/src/openpi/models/pi0.py:47-63`。

4. pi0 主线把 timestep embedding broadcast 到每个 action token，并与 action embedding 拼接后过 MLP：
   - `openpi/src/openpi/models/pi0.py:171-177`：
     - `time_tokens = repeat(time_emb, s=action_horizon)`
     - `action_time_tokens = concat([action_tokens, time_tokens], axis=-1)`
     - `action_time_mlp_in -> swish -> action_time_mlp_out`
   - 这对应论文附录中的 timestep-action 融合：

$$
W_3\,\mathrm{swish}\!\left(W_2\,\mathrm{concat}(W_1 a^\tau,\phi(\tau))\right)
$$

   - 代码位置见论文 `openpi/paper/Black_等_-_2026_-_π_0_A_Vision-Language-Action_Flow_Model_for_General_Robot_Control.md:415`。

5. noisy action tokens 的 attention mask：
   - `openpi/src/openpi/models/pi0.py:179-185`：把 action tokens 拼到 suffix 后，并追加 `ar_mask += [True] + [False] * (H-1)`。
   - 这表示第一个 action token 开启新 block，剩下 $H-1$ 个 action token 与它同 block；所以所有 action tokens 内部双向 attention，并能 attend prefix 与 state，但 prefix/state 不能 attend 到 action block。

从这一步可以推出：state 与 noisy actions 作为 `suffix_tokens` 进入 expert 1，也就是 action expert。调用点同样是 `self.PaliGemma.llm([prefix_tokens, suffix_tokens], ...)` 的第 1 个元素，见 `openpi/src/openpi/models/pi0.py:209-211`。

PyTorch 对照：`openpi/src/openpi/models_pytorch/pi0_pytorch.py:238-315`。

## 6. Attention mask 如何精确实现三块信息流

核心函数是 `make_attn_mask(input_mask, mask_ar)`：

- `openpi/src/openpi/models/pi0.py:19-44`：先对 `mask_ar` 做 cumulative sum；若 key token 的累计 block id 小于等于 query token 的累计 block id，则 query 可以 attend 该 key；最后再与 padding validity mask 相与。

结合 prefix/suffix 的 `ar_mask`：

- prefix block：$[\mathrm{images},\mathrm{language}]$，`ar_mask` 全 `False`。
- state block：$[\mathrm{state}]$，`ar_mask` 第一个为 `True`。
- action block：$[a_0^\tau,\ldots,a_{H-1}^\tau]$，`ar_mask` 第一个为 `True`，其余为 `False`。

因此完整信息流为：

- images/language query：只能看 images/language。
- state query：能看 images/language + state。
- action query：能看 images/language + state + 所有 action tokens。

训练时：

- `openpi/src/openpi/models/pi0.py:203-208`：分别得到 prefix/suffix tokens 和 masks，拼成总 `input_mask` 与 `ar_mask`，再用 `make_attn_mask` 得到 attention mask。
- `openpi/src/openpi/models/pi0.py:209-211`：一次性把 `[prefix_tokens, suffix_tokens]` 输入 LLM。

推理时：

- `openpi/src/openpi/models/pi0.py:233-237`：先只跑 prefix，填充 `kv_cache`。
- `openpi/src/openpi/models/pi0.py:241-252`：每个 denoising step 只嵌入 suffix，构造 suffix 内部 mask，并拼接一个 suffix-to-prefix mask，使 suffix query 可以 attend prefix cache。
- `openpi/src/openpi/models/pi0.py:261-267`：用 `kv_cache=kv_cache` 只计算 suffix 的 transformer 输出。

这与论文附录一致：state block 因不随 flow step 改变而可缓存，action block 在每一步 flow integration 中重复前向；见论文 `openpi/paper/Black_等_-_2026_-_π_0_A_Vision-Language-Action_Flow_Model_for_General_Robot_Control.md:417` 与 `openpi/paper/Black_等_-_2026_-_π_0_A_Vision-Language-Action_Flow_Model_for_General_Robot_Control.md:433`。

## 7. VLM 与 Action Expert 在 Gemma 内部如何交互

Gemma 模块本身支持多个 expert：

- `openpi/src/openpi/models/gemma.py:340-343`：`Module` 注释说明它是“supporting a mixture of different weights for different tokens”，`configs` 是每个 expert 的 config 列表。
- `openpi/src/openpi/models/gemma.py:389-399`：`__call__` 接收 `embedded: Sequence[...]`，也就是每个 expert 一组 token embedding。
- `openpi/src/openpi/models/gemma.py:443-450`：命名规则说明 expert 0 使用无后缀参数名以便加载 PaliGemma checkpoint，后续 expert 用 `_1` 后缀并从头初始化；实践中就是 PaliGemma 与 action expert。

在每个 transformer block 内：

1. 每个 expert 各自做 RMSNorm：
   - `openpi/src/openpi/models/gemma.py:297-305`。

2. attention 是跨 expert 交互的唯一位置：
   - `openpi/src/openpi/models/gemma.py:172-201`：对每个 expert 分别用自己的 Q/K/V 权重生成 qkv，然后 `jnp.concatenate(..., axis=1)` 沿序列维拼接所有 expert 的 q/k/v。
   - `openpi/src/openpi/models/gemma.py:216-230`：在拼接后的全序列上计算 attention logits、mask、softmax 和 value 聚合。
   - `openpi/src/openpi/models/gemma.py:233-249`：再按原 expert 的 token 长度把 attention 结果切回各 expert，并用各自的 output projection。

3. FFN 是 expert-local：
   - `openpi/src/openpi/models/gemma.py:314-324`：对每个 expert 用自己的 `pre_ffw_norm_i` 和 `mlp_i`。

所以，“VLM 与 action expert 的信息流”在代码中的精确定义是：

- 图像/语言 token 走 expert 0 的 embedding/attention projection/FFN 参数。
- state/noisy action token 走 expert 1 的 state/action projection、attention projection、FFN 参数。
- 二者只在 self-attention 的 K/V/Q 混合中交换信息；action token 作为 query 可以 attend 图像/语言/state/action 的 K/V，VLM prefix token 由于 attention mask 不能 attend state/action。

这直接对应论文“weights interact only through self-attention layers”的描述，见论文 `openpi/paper/Black_等_-_2026_-_π_0_A_Vision-Language-Action_Flow_Model_for_General_Robot_Control.md:419`。

## 8. 训练 loss：从真实 action chunk 到 flow matching MSE

训练循环入口：

- `openpi/scripts/train.py:136-151`：`train_step` 内的 `loss_fn` 调用 `model.compute_loss(rng, observation, actions, train=True)`，然后对返回的 chunk loss 做 `jnp.mean`。
- `openpi/scripts/train.py:153-158`：用 step-folded rng 计算 loss 与梯度。

`Pi0.compute_loss` 的细节：

1. 预处理 observation：
   - `openpi/src/openpi/models/pi0.py:192-194`：拆出 preprocess/noise/time rng，并调用 `preprocess_observation(..., train=train)`。
   - `preprocess_observation` 会检查图像 key、resize 到 `224x224`，训练时对图像做 random crop/resize/rotate/color jitter，再补齐 image mask；见 `openpi/src/openpi/models/model.py:144-208`。

2. 采样 flow noise 和 timestep：
   - `openpi/src/openpi/models/pi0.py:195-198`：
     - $\mathrm{noise}\sim\mathcal{N}(0,I)$，形状与 `actions` 相同。
     - $\mathrm{time}\sim 0.999\cdot\mathrm{Beta}(1.5,1)+0.001$。
   - 这对应论文附录“timestep 采样使用强调低 timestep 的 beta 分布、阈值 s=0.999”，见论文 `openpi/paper/Black_等_-_2026_-_π_0_A_Vision-Language-Action_Flow_Model_for_General_Robot_Control.md:421`。

3. 构造 noisy action 与目标向量场：
   - `openpi/src/openpi/models/pi0.py:199-200`：

$$
\begin{aligned}
x_t &= \mathrm{time}\cdot\mathrm{noise} + (1-\mathrm{time})\cdot\mathrm{actions}, \\
u_t &= \mathrm{noise}-\mathrm{actions}.
\end{aligned}
$$

   - 代码注释在推理处明确说明当前实现采用 diffusion 文献中常见约定：$t=1$ 是 noise，$t=0$ 是 target，且“opposite of the pi0 paper”；见 `openpi/src/openpi/models/pi0.py:225-228`。因此当前代码里的 `time` 与论文符号 $\tau$ 方向相反，但训练目标自洽：模型学习从当前 $x_t$ 沿 $u_t=\mathrm{noise}-\mathrm{actions}$ 的方向场，推理时用负步长从 noise 积分到 action。

4. 嵌入 prefix 与 suffix：
   - `openpi/src/openpi/models/pi0.py:202-207`：调用 `embed_prefix(observation)` 与 `embed_suffix(observation, x_t, time)`，拼接 mask。

5. transformer 前向并只取 action token 输出：
   - `openpi/src/openpi/models/pi0.py:208-211`：计算 positions，一次性前向 `[prefix_tokens, suffix_tokens]`。
   - `openpi/src/openpi/models/pi0.py:212`：`suffix_out[:, -self.action_horizon:]` 只取最后 $H$ 个 noisy action tokens 的输出，再经 `action_out_proj` 得到 $v_t$，形状 $[B,H,\mathrm{action\_dim}]$。

6. loss 是 action 维上的 MSE，保留 batch 与 horizon：
   - `openpi/src/openpi/models/pi0.py:214`：`return mean(square(v_t - u_t), axis=-1)`，即 $\frac{1}{D}\sum_{d=1}^{D}(v_{t,d}-u_{t,d})^2$，返回 $[B,H]$。
   - 训练脚本再在 `openpi/scripts/train.py:150-151` 对 $[B,H]$ 做全局 mean，得到标量 loss。

PyTorch 对照：

- `openpi/src/openpi/models_pytorch/pi0_pytorch.py:317-374` 完成同样流程，区别是最后 `F.mse_loss(u_t, v_t, reduction="none")` 保留 $[B,H,D]$，训练脚本在 `openpi/scripts/train_pytorch.py:529` 调用模型后再聚合。

## 9. 推理：从 observation 采样 action chunk

推理入口是 policy：

- `openpi/src/openpi/policies/policy_config.py:75-90`：创建 policy 时，input transform 顺序为 repack、default prompt、data transform、normalize、model transform；output transform 顺序为 model output、unnormalize、data output、repack output。
- `openpi/src/openpi/policies/policy.py:67-90`：`infer` 复制 obs，应用 input transform，加 batch 维，构造 `Observation.from_dict(inputs)`。
- `openpi/src/openpi/policies/policy.py:91-105`：调用 `self._sample_actions(..., observation, **sample_kwargs)`，输出 `actions` 后去 batch 维并执行 output transform。

`Pi0.sample_actions` 的采样流程：

1. observation 预处理：
   - `openpi/src/openpi/models/pi0.py:225`：`preprocess_observation(None, observation, train=False)`。推理不做随机增强，但会 resize 和补 mask。

2. 初始化为高斯噪声：
   - `openpi/src/openpi/models/pi0.py:228-232`：默认 `num_steps=10`，$dt=-1/\mathrm{num\_steps}$，如果没有外部 noise，则采样 $[B,H,D]$ 高斯噪声。

3. prefix cache：
   - `openpi/src/openpi/models/pi0.py:233-237`：只嵌入图像/语言 prefix，构造 prefix attention mask，跑 LLM 得到 `kv_cache`。

4. 每个 flow step 只跑 suffix/action expert：
   - `openpi/src/openpi/models/pi0.py:239-271` 定义单步：
     - 用当前 $x_t$ 和当前 $\mathrm{time}$ 调 `embed_suffix`。
     - 构造 suffix 内部 mask 与 suffix-to-prefix mask。
     - 用 prefix `kv_cache` 前向 `[None, suffix_tokens]`。
     - 取最后 $H$ 个 action token 输出，经 `action_out_proj` 得到 $v_t$。
     - Euler 更新：$x_t \leftarrow x_t + dt\cdot v_t$，$\mathrm{time}\leftarrow\mathrm{time}+dt$。

5. 循环从 $\mathrm{time}=1.0$ 到 $0$：
   - `openpi/src/openpi/models/pi0.py:273-278`：`while_loop` 直到 `time >= -dt/2` 不再满足。
   - 返回 $x_0$，即 action chunk。

PyTorch 对照：

- prefix cache：`openpi/src/openpi/models_pytorch/pi0_pytorch.py:386-400`。
- denoising loop：`openpi/src/openpi/models_pytorch/pi0_pytorch.py:402-420`。
- suffix denoise step：`openpi/src/openpi/models_pytorch/pi0_pytorch.py:422-462`。

## 10. 端到端信息流摘要

下面用一条可直接对应代码的链路概括：

1. 原始机器人样本。
2. dataset 按 `action_horizon` 取未来 $H$ 步动作。
3. repack 成 `image/state/actions/prompt`。
4. embodiment transform 统一到三路图像 + state + actions + prompt。
5. `Normalize(state, actions)`。
6. `TokenizePrompt(prompt)` 得到 `tokenized_prompt`。
7. `PadStatesAndActions` 到 `action_dim`。
8. `Observation.from_dict / Actions`。
9. `preprocess_observation` 执行 resize、augment、mask。
10. `embed_prefix`：images $\rightarrow$ SigLIP image tokens；prompt ids $\rightarrow$ PaliGemma token embedding；路由到 expert 0 / VLM backbone。
11. `compute_loss` 训练时：$\mathrm{actions}+\mathrm{noise}+\mathrm{time}\rightarrow x_t$。
12. `embed_suffix`：state $\rightarrow$ `state_proj` $\rightarrow$ state token；noisy actions $\rightarrow$ `action_in_proj`；time $\rightarrow$ sincos $\rightarrow$ concat(action,time) $\rightarrow$ MLP；路由到 expert 1 / action expert。
13. `make_attn_mask`：prefix 只能看 prefix；state 看 prefix + state；action 看 prefix + state + action。
14. Gemma two-expert transformer：expert-local RMSNorm/FFN，cross-expert self-attention 交换信息。
15. 只取最后 $H$ 个 action token 输出，经 `action_out_proj` 得到 $v_t$。
16. loss：$\mathrm{mean}_D\left((v_t-(\mathrm{noise}-\mathrm{actions}))^2\right)$。
17. `train.py` 再对 $B,H$ 维度求 mean 得到标量。

推理时把中间 $x_t=\mathrm{time}\cdot\mathrm{noise}+(1-\mathrm{time})\cdot\mathrm{actions}$ 换成：

$$
\begin{aligned}
x_1 &\sim \mathcal{N}(0,I), \\
v_t &= \mathrm{model}(x_t,\mathrm{observation},t), \\
x_{t+dt} &= x_t + dt\cdot v_t,\qquad dt=-1/\mathrm{num\_steps}, \\
\mathrm{return}\quad x_0 &\quad \text{as action chunk}.
\end{aligned}
$$

## 11. 最关键代码位置索引

- 输入数据结构：`openpi/src/openpi/models/model.py:50-74`、`openpi/src/openpi/models/model.py:81-141`
- pi0 config 与 shape spec：`openpi/src/openpi/models/pi0_config.py:18-86`
- 数据 transform pipeline：`openpi/src/openpi/training/data_loader.py:172-191`、`openpi/src/openpi/training/config.py:113-125`
- prompt tokenizer：`openpi/src/openpi/models/tokenizer.py:14-48`
- 图像/语言 prefix embedding：`openpi/src/openpi/models/pi0.py:105-137`
- state/action/time suffix embedding：`openpi/src/openpi/models/pi0.py:139-186`
- blockwise attention mask：`openpi/src/openpi/models/pi0.py:19-44`
- loss：`openpi/src/openpi/models/pi0.py:188-214`
- sampling：`openpi/src/openpi/models/pi0.py:216-279`
- two-expert Gemma：`openpi/src/openpi/models/gemma.py:340-411`
- cross-expert attention 拼接与拆分：`openpi/src/openpi/models/gemma.py:172-249`
- train step 聚合 loss：`openpi/scripts/train.py:136-158`
- inference policy wrapper：`openpi/src/openpi/policies/policy.py:67-106`
