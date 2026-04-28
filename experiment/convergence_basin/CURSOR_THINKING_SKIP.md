# Cursor「Thinking / Skipped」卡顿 — 可复制 Prompt

与 IDE 索引、`composer diff`、上下文过大有关；本仓库已通过 `.cursorignore` 排除大块数据与缓存。仍卡时可粘贴下文。

---

### 1. 强制放弃旧 diff、直接重写核心逻辑

```
Ignore all current context analysis. DO NOT try to diff with existing code. JUST REWRITE the entire core logic of [文件名] according to the latest requirements. Minimize text explanation, just output the full code block.
```

### 2. 上下文过载 — 锁死视野

```
Strictly focus on the files attached to this chat. IGNORE the rest of the workspace and all binary/data files. Re-initialize your reasoning only based on the provided code logic.
```

### 3. 逻辑死循环 / 反复 Skipped

```
Reset your internal state for this task. You are currently stuck in a logic loop. Let's start fresh: Apply P_pred_w = (P_pred_cano · S + C) ∘ T_coarse to the pose estimation script (row-vector: p_w = p_obj @ R + t). No more skipping.
```

### 4. 自检（暴力排错）

```
Diagnostic Run: You are taking too long to think/skip. Is it because of large data files in context, or complex diff calculation? If so, stop what you are doing, acknowledge the issue, and provide a stripped-down version of the code in plain text format instead of the composer diff.
```

---

另见：`.cursor/rules/avoid-thinking-stall.mdc`。
