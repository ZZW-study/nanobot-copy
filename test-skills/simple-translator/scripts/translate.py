#!/usr/bin/env python3
"""
简易翻译脚本 - 使用 MyMemory 免费API

用法:
    python translate.py "要翻译的文本" zh en
    python translate.py "Hello" en zh

参数:
    text: 要翻译的文本
    source: 源语言代码 (zh, en, ja, ko, fr, de, es)
    target: 目标语言代码
"""

import sys
import urllib.parse
import urllib.request
import json


def translate(text: str, source: str = "zh", target: str = "en") -> str:
    """
    翻译文本

    Args:
        text: 要翻译的文本
        source: 源语言代码
        target: 目标语言代码

    Returns:
        翻译后的文本
    """
    # URL编码文本
    encoded_text = urllib.parse.quote(text)

    # 构建API URL
    url = f"https://api.mymemory.translated.net/get?q={encoded_text}&langpair={source}|{target}"

    try:
        # 发送请求
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))

            if "responseData" in data and "translatedText" in data["responseData"]:
                return data["responseData"]["translatedText"]
            else:
                return f"翻译失败: {data.get('error', '未知错误')}"

    except urllib.error.URLError as e:
        return f"网络错误: {e}"
    except json.JSONDecodeError:
        return "解析响应失败"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\n示例:")
        print('  python translate.py "你好" zh en')
        print('  python translate.py "Hello" en zh')
        sys.exit(1)

    text = sys.argv[1]
    source = sys.argv[2] if len(sys.argv) > 2 else "zh"
    target = sys.argv[3] if len(sys.argv) > 3 else "en"

    result = translate(text, source, target)
    print(result)


if __name__ == "__main__":
    main()
