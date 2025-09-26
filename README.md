# ComfyUI 节点：Gemini 香蕉助手 by ai来事

一个面向生产的 ComfyUI 节点，使用 Gemini 2.5 Flash Image（兼容 API）对输入图像进行编辑。将本项目放入 `ComfyUI/custom_nodes/` 后，即可在 ComfyUI 中用自然语言指令进行图像修改。
> #### 调用【KUAI(酷爱) API】方法请查看：https://docs.kuai.host/7451843m0

![alt text](show.jpg)

## 环境要求
- ComfyUI
- Python 环境安装依赖：numpy、pillow、requests、torch（ComfyUI 环境通常已含）
- 可用的 Gemini 兼容 API Key（支持自定义 Base URL） 默认为 `https://apis.kuai.host/`

## 安装
1. 复制或克隆本仓库到 `ComfyUI/custom_nodes/confyui-gemini-nano-banana`。
2. 在 ComfyUI 使用的 Python 环境中执行：`pip install -r requirements.txt`（一般都有了，无需安装）。
3. 重启 ComfyUI。节点显示为“AI / Gemini -> Gemini 香蕉助手 by ai来事”。

## 节点输出
- image：编辑后的图像张量（保持输入布局：BHWC 或 BCHW）
- metrics_json：JSON 字符串，包含延迟、重试次数、请求次数、token 使用等指标

## 参数与行为

### 必填
- image：输入图像张数或 PIL Image。批大小必须为 1。
- prompt：自然语言说明如何编辑图像。节点会限制最大 1000 字符。
- api_key：Gemini 兼容 API 的 Key。
- timeout：HTTP 请求超时时间（默认为30秒）。

### 可选
- instruction_preset：从 `instruction_presets/` 选择 JSON 预设（下拉出现相对路径，如 `enhanced/color-flat.json`）。用于追加模板化指令或系统说明，帮助风格/安全/一致性。
- system_prompt：从 `instruction_presets/system_prompt/` 选择系统提示词（Markdown）。作为系统级指令影响整体行为；可留空以保持中性。
- image_2 / image_3 / image_4：额外参考图像，帮助模型将角色/道具/细节融合到主图。
- max_output_tokens：响应 token 上限（控制模型“说话”量与成本；0 用服务默认）。
- minimal_response：启用后，在提示中追加“仅返回编辑后 PNG，不要文字说明”的提醒，减小模型返回文本的几率。
- edit_strength：编辑强度提示（0=保守，1=强烈）。
- color_lock：鼓励保留原始配色（品牌素材常用）。
- style_lock：鼓励保持原始画风/笔触。
- temperature：采样温度覆写（提高多样性，接近 0 更稳定）。
- top_p：核采样上限（降低可能减少幻觉但牺牲多样性）。
- base_url：Gemini 兼容服务基础地址，默认 `https://apis.kuai.host/`（与官方接口格式一致即兼容）。

## 工作流建议
- 基本流程：`Load Image` -> `Gemini 香蕉助手 by ai来事` -> `Save Image`
- 初次使用建议将 temperature/top_p 设为 0，minimal_response 保持启用，先验证基础行为
- 需要强调风格或规则时，使用 instruction_preset 与 system_prompt 管理通用指令；参考图用于自然融合
- 如果输出没有图像或被安全阻断：精简提示词；明确要求“仅返回编辑后的 PNG 图像”

## 指令预设（instruction_presets）
- 位置：`instruction_presets/`（支持子目录）
- 作用：将模板化的指令片段与用户提示合并，减少重复书写，提升一致性
- JSON 预设字段示例：
  ```json
  {
    "name": "显示名称（可选）",
    "description": "预设作用说明",
    "system_prompt": "系统级指导（与节点 system_prompt 叠加）",
    "prompt_prefix": "在用户提示前插入",
    "directives": ["额外句子或要点", "另一条指令"],
    "prompt_suffix": "在所有指令后追加",
    "extra_parts": ["额外的用户消息部分（文本）"]
  }
  ```
- 示例（本仓库）：
  - enhanced/character-cutout.json：主角抠图，透明背景，强调不引入新内容
  - enhanced/detail-boost.json：细节增强与微对比，避免光晕与伪影

## 系统提示词（system_prompt）
- 位置：`instruction_presets/system_prompt/`（Markdown）
- 作用：作为“系统指令”提交，塑造整体行为与偏好；适合品牌合规、最小改动策略、多图合成规则等场景
- 是否必要：非必须。留空可获得中性行为；按项目需求增删文件即可
- 示例（本仓库）：
  - brand-style-guardian.md：保持品牌风格与配色一致
  - minimal-edit.md：仅按用户请求进行最小改动
  - multi-image-composer.md：多图融合时遵循光照/透视/尺度的一致性

## 网络与日志
- 节点调用端点：`{base_url}/v1beta/models/{model}:generateContent`（默认 base_url 为 `https://apis.kuai.host/`）
- 针对 429/5xx 的指数退避重试（默认最多 2 次），timeout 覆盖每次尝试
- 日志包含打码的 API key 与指标，不写入明文 Key

## 故障排查
- 没有图像输出/被安全阻断：精简提示词；启用 minimal_response；明确“仅返回 PNG 图像”
- 幻觉（多手/多物体）：降低 temperature 与 top_p；避免堆叠冲突指令；减少系统提示
- 频繁限速：增大 timeout；降低短时间内重复运行；适度限制 max_output_tokens
- 依赖缺失（类型检查器报错）：在运行环境安装 numpy、pillow、requests、torch
