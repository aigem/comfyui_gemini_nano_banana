【 获取API Key:https://apis.kuai.host 低至 0.05x/次】

Gemini 文生图
- 功能：文本生成图像
- 接口：{base_url}/v1beta/models/{model}:generateContent（默认 base_url=https://apis.kuai.host/，model=gemini-2.5-flash-image-preview）

主要输入
- prompt（必填）：详细描述希望生成的内容/风格/细节（建议 ≤1000 字符）。
- instruction_preset / system_prompt：集中管理风格与规则，便于团队复用与迭代。
- minimal_response：建议启用，要求仅返回 PNG 图像。
- temperature/top_p/max_output_tokens：生成采样与长度控制。
- num_outputs/parallel：批量并行生成多张图。

输出
- image：生成的图像张量（BHWC）。
- metrics_json：请求耗时与 token 指标。

使用建议
- 风格一致性：结合 instruction_presets/system_prompt/brand-style-guardian.md。
- 严格控制口径：通过系统提示集中约束伦理/品牌/禁忌词。
- 若返回文本多于图像：启用 minimal_response，并在提示中显式要求“只返回 PNG”。