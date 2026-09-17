Generate a concise title for the conversation. Output only the title, no quotes or explanations.

Rules:
1. Use the conversation's dominant language.
2. Concrete, not abstract: keep domain words, file/feature names, or the broken behavior. Someone who wasn't there should still recognize the topic.
3. Bug fixes start with 修复 / Fix.
4. Noun or verb phrase, at most 30 characters, no trailing punctuation.

Examples:

Input: 修复一个 "/bg" 相关的问题。为什么一个terminal，只挂了一个session，还要确认？
Good: 修复 /bg 挂起会话恢复时的多余确认
Bad:  单会话仍要确认      <- too abstract, loses 修复 and /bg
Bad:  修复一个问题        <- too generic

Input: (一张截图) 帮我看看这个报错，我的 redis 连不上
Good: 排查 Redis 连接报错  <- from the accompanying text, not "看截图"
Bad:  图片问题

Input: why does resuming a single /bg session still ask for confirmation?
Good: Fix /bg double-confirm
Bad:  session confirm      <- too abstract, loses /bg
