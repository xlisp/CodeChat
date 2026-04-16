## 手动加载这个模型checkpoints/codechat_8b_rl_funcall_v5/step_000060.pt 测试使用

```
⏺ 三种用法

  1. One-shot 最快验证

  python -m scripts.funcall_cli \
      --ckpt checkpoints/codechat_8b_rl_funcall_v5/step_000060.pt \
      --user "What's the weather in Paris?"

  默认内置一个 get_weather 工具，直接出结果：

  --- raw output ---
  <functioncall> {"name": "get_weather", "arguments": {"location": "Paris"}}

  --- parsed call ---
    name:      get_weather
    arguments: {"location": "Paris"}

  2. 自定义工具（文件）

  建个 tools.json：

  [
    {
      "name": "search_restaurant",
      "description": "Find restaurants near a location",
      "parameters": {
        "type": "object",
        "properties": {
          "city": {"type": "string"},
          "cuisine": {"type": "string"}
        },
        "required": ["city"]
      }
    }
  ]

  python -m scripts.funcall_cli \
      --ckpt checkpoints/codechat_8b_rl_funcall_v5/step_000060.pt \
      --tools-file tools.json \
      --user "找一家北京的川菜馆"

  3. REPL 多轮（含 function response 回灌）

  python -m scripts.funcall_cli \
      --ckpt checkpoints/codechat_8b_rl_funcall_v5/step_000060.pt \
      --tools-file tools.json

  交互流程：
  - user> 输入请求 → 模型 emit <functioncall>
  - function_response> 这里粘你工具的返回 JSON（或直接回车跳过）
  - 继续下一轮 user

  Python 里程序化调用（最小片段）

  from scripts.funcall_cli import load_model, build_system, run_once, parse_call

  model, cfg = load_model("checkpoints/codechat_8b_rl_funcall_v5/step_000060.pt")
  system_text = build_system([{
      "name": "get_weather",
      "description": "...",
      "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
  }])
  raw = run_once(model, cfg,
                 system_text,
                 [("user", "Weather in Tokyo?")],
                 max_new_tokens=256, temperature=0.7, top_k=50)
  call = parse_call(raw)       # {"name": "get_weather", "arguments": {"location": "Tokyo"}}

  关键点

  - prompt 格式必须带 <|system|> + 工具 schema——这是 SFT 数据的格式。直接用 chat_cli.py 聊天的话，模型拿不到工具定义，会退化成普通对话。
  - parse_call 用的是和 reward 同一套宽松 JSON 解析（_parse_json_loose + _unwrap_args），能兜住单引号/双引号混用那种情况。
  - 默认 temperature=0.7（比评测时的 1.0 低），推理时更稳；想要 Pass@16 那种多样性采样就调到 1.0。
  - 长 prompt 会自动左截断（跟训练一致），不会因为工具 schema 太大崩掉。

```
