---
name: simple-translator
description: 翻译文本到指定语言。支持中英日韩等多语言互译。当用户要求翻译、转换语言、或说"用英语怎么说"时使用。
---

# 简易翻译器

使用 MyMemory 免费翻译API进行文本翻译，无需API密钥。

## 快速开始

翻译文本到目标语言：

```bash
curl -s "https://api.mymemory.translated.net/get?q=你好&langpair=zh|en" | jq -r '.responseData.translatedText'
```

输出：`Hello`

## 支持的语言

| 代码 | 语言 |
|------|------|
| zh | 中文 |
| en | 英语 |
| ja | 日语 |
| ko | 韩语 |
| fr | 法语 |
| de | 德语 |
| es | 西班牙语 |

## 使用方法

### 基本翻译

```bash
# 中文 → 英语
curl -s "https://api.mymemory.translated.net/get?q=你好世界&langpair=zh|en" | jq -r '.responseData.translatedText'

# 英语 → 中文
curl -s "https://api.mymemory.translated.net/get?q=Hello World&langpair=en|zh" | jq -r '.responseData.translatedText'

# 中文 → 日语
curl -s "https://api.mymemory.translated.net/get?q=谢谢&langpair=zh|ja" | jq -r '.responseData.translatedText'
```

### 处理空格

URL中的空格需要替换为 `%20` 或使用 `+`：

```bash
# 方法1：使用 +
curl -s "https://api.mymemory.translated.net/get?q=Hello+World&langpair=en|zh"

# 方法2：使用 URL 编码
curl -s "https://api.mymemory.translated.net/get?q=Hello%20World&langpair=en|zh"
```

### 获取完整响应

```bash
curl -s "https://api.mymemory.translated.net/get?q=你好&langpair=zh|en" | jq .
```

响应示例：
```json
{
  "responseData": {
    "translatedText": "Hello",
    "match": 1
  },
  "quotaFinished": false
}
```

## 注意事项

- 免费API有调用限制（每天约1000次）
- 翻译质量适合日常使用，专业场景建议使用付费服务
- 长文本建议分段翻译
