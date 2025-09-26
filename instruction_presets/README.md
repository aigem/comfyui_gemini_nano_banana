# Instruction Presets

This directory bundles reusable prompt templates for the `Gemini Edit Image` node. Drop additional JSON files here to expose them in the node's `instruction_preset` dropdown.

## Adding a new preset
1. Create a JSON file with the schema below. Nested folders are allowed.
2. Keep names short and descriptive; nested paths appear in the UI using `/` separators.

```json
{
  "name": "Optional label shown in the UI",
  "description": "Short explanation of what the preset does",
  "system_prompt": "Optional higher-level guidance",
  "prompt_prefix": "Text placed before the user prompt",
  "directives": ["Bullet directives", "Sent to Gemini as individual lines"],
  "prompt_suffix": "Text appended after directives",
  "extra_parts": ["Optional secondary user messages"]
}
```

Only include keys you need; the node gracefully skips missing fields. Keep sensitive workflow details out of these files since they ship with the node.
