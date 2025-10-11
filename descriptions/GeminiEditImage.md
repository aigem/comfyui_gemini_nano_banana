【 获取API Key:https://apis.kuai.host 低至 0.05x/次】

Gemini 香蕉助手（图生图/多图生图）
- 功能：使用 Gemini 2.5 Flash Image（兼容 API）基于输入图像进行编辑，可附加多张参考图。
- 接口：{base_url}/v1beta/models/{model}:generateContent（默认 base_url=https://apis.kuai.host/，model=gemini-2.5-flash-image）

主要输入
- image（必填）：主图像。
- image_2/3/4（可选）：参考图像。
- prompt（必填）：编辑指令（建议 ≤1000 字符）。
- instruction_preset / system_prompt：从仓库 instruction_presets/ 选择模板与系统提示，统一风格与规则。
- minimal_response：建议启用，以减少返回文本，专注返回 PNG 图像。
- temperature/top_p/max_output_tokens：生成采样与长度控制。
- num_outputs/parallel：批量并行生成多张图。

输出
- image：编辑后的图像张量（保持输入维度布局）。
- metrics_json：请求耗时、重试次数、token 使用等指标。

使用建议
- 初次使用：将 temperature/top_p 设为 0，启用 minimal_response。
- 若无图像返回或被安全阻断：精简/明确提示；强调“仅返回 PNG 图像”。
- 多图融合：搭配 system_prompt/multi-image-composer.md，统一光照/透视/尺度。