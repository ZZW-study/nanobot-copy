# 🤖 ZBot - 你的个人 AI 命令行助手

ZBot 是一个强大的命令行 AI 助手，支持多模型切换、工具调用、定时任务、MCP 扩展等功能。

## ✨ 功能特性

### 💬 智能对话
- **多模型支持**：无缝切换 OpenAI、DeepSeek、阿里通义千问、硅基流动等模型
- **对话历史**：自动保存对话上下文，支持多会话管理
- **Markdown 渲染**：优雅地显示代码、表格等格式化内容
- **网关支持**：兼容 OpenRouter 等聚合平台

### 🛠️ 工具调用
- **Shell 命令**：AI 可以帮你执行系统命令
- **文件操作**：读写、搜索、管理工作区文件
- **网页搜索**：联网搜索获取最新信息
- **网页内容提取**：智能提取网页正文

### ⏰ 定时任务
- **自然语言创建**：用日常语言描述即可创建定时任务
- **执行历史记录**：查看任务执行结果和日志
- **灵活调度**：支持 Cron 表达式

### 🔌 MCP 扩展
- **协议兼容**：支持 Model Context Protocol (MCP) 标准
- **动态加载**：运行时连接多个 MCP 服务器
- **工具扩展**：接入官方或社区提供的 MCP 工具

### 🛡️ 安全控制
- **工作区限制**：AI 只能访问指定目录
- **超时控制**：防止命令无限执行
- **代理配置**：支持 HTTP/SOCKS 代理

## 🚀 快速开始

### 环境要求

- Python 3.11+
- 支持的系统：Windows、macOS、Linux

### 1. 安装 ZBot

```bash
# 从源码安装
cd ZBot
pip install -e .

# 或使用 Docker（见下文）
```

### 2. 初始化配置

```bash
# 首次运行，自动创建配置文件
python -m ZBot onboard
```

### 3. 配置模型

编辑生成的配置文件 `~/.ZBot/config.json`：

```json
{
  "model": "deepseek/deepseek-chat",
  "providers": {
    "deepseek": {
      "api_key": "your_deepseek_api_key",
      "api_base": "https://api.deepseek.com/v1"
    }
  },
  "workspace": "~/.ZBot/workspace"
}
```

**常用模型配置参考**：

| 提供商 | 模型名称 | API Base |
|--------|----------|----------|
| DeepSeek | `deepseek/deepseek-chat` | `https://api.deepseek.com/v1` |
| 阿里通义 | `dashscope/qwen-turbo` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| 硅基流动 | `siliconflow/DeepSeek-V2.5` | `https://api.siliconflow.cn/v1` |
| OpenAI | `openai/gpt-4o-mini` | `https://api.openai.com/v1` |
| OpenRouter | `openrouter/anthropic/claude-3-haiku` | `https://openrouter.ai/api/v1` |

### 4. 开始对话

```bash
# 交互模式
python -m ZBot agent

# 单次对话
python -m ZBot agent -m "你好，请帮我写一个 Python 快速排序"

# 指定会话
python -m ZBot agent -s "work" -m "总结今天的会议要点"

# 查看配置状态
python -m ZBot status
```

## 📖 使用指南

### 交互模式命令

```
你：帮我查询今天的天气
🤖 ZBot 正在思考...
↳ 正在调用工具：网页搜索
↳ 进度：天气查询完成
ZBot
今天北京天气：晴转多云，22-30℃，空气质量优。

你：/new           # 开始新会话
你：exit           # 退出程序
```

### 创建定时任务

```bash
# 每天早上 9 点提醒我写日报
python -m ZBot agent -m "每天早上 9 点提醒我写日报"

# 每周一检查项目更新
python -m ZBot agent -m "每周一早上 10 点检查 GitHub 项目更新"
```

### 配置 MCP 服务器

在 `config.json` 中添加：

```json
{
  "tools": {
    "mcp_servers": {
      "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"]
      },
      "github": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"]
      }
    }
  }
}
```

## 🏗️ 项目架构

```
ZBot/
├── ZBot/
│   ├── __main__.py          # CLI 入口
│   ├── __init__.py          # 包元信息
│   ├── cli/
│   │   └── commands.py      # Typer 命令行命令
│   ├── agent/               # Agent 核心
│   │   ├── loop.py          # Agent 执行循环
│   │   ├── context.py       # 上下文构建
│   │   ├── memory.py        # 记忆管理
│   │   ├── skills.py        # 技能加载
│   │   └── tools/           # 工具实现
│   │       ├── base.py      # 工具基类
│   │       ├── shell.py     # Shell 命令
│   │       ├── filesystem.py # 文件操作
│   │       ├── web.py       # 网页搜索
│   │       ├── cron.py      # 定时任务
│   │       ├── mcp.py       # MCP 集成
│   │       └── registry.py  # 工具注册表
│   ├── config/              # 配置模块
│   │   ├── schema.py        # 配置结构定义
│   │   ├── loader.py        # 配置加载器
│   │   └── paths.py         # 路径工具
│   ├── providers/           # LLM 提供商
│   │   ├── base.py          # 提供商基类
│   │   ├── litellm_provider.py  # LiteLLM 实现
│   │   └── registry.py      # 提供商注册表
│   ├── session/             # 会话管理
│   │   └── manager.py       # 会话管理器
│   ├── cron/                # 定时任务
│   │   ├── service.py       # 任务服务
│   │   └── types.py         # 类型定义
│   ├── skills/              # 技能模块
│   │   └── skill-creator/   # 技能创建工具
│   ├── templates/           # 模板文件
│   └── utils/               # 工具函数
├── Dockerfile               # Docker 构建
├── docker-compose.yml       # Docker Compose
└── requirements.txt         # Python 依赖
```

## 🐳 Docker 部署

```bash
# 交互模式运行
docker-compose run --rm zbot

# 或启动容器后进入交互
docker-compose up -d
docker exec -it zbot python -m ZBot agent
```

## 📦 依赖说明

```
# CLI 框架
typer>=0.9.0              # 命令行框架
prompt_toolkit>=3.0.0     # 高级终端输入
rich>=13.0.0              # 富文本输出

# LLM 接口
litellm>=1.0.0            # 多模型统一接口

# 数据验证
pydantic>=2.0.0           # 数据模型
pydantic-settings>=2.0.0  # 配置管理

# MCP 支持
mcp>=1.0.0                # Model Context Protocol

# 可选依赖
readability-lxml>=0.8.0   # 网页内容提取
```

## ⚙️ 配置选项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `model` | `""` | 使用的模型名称 |
| `workspace` | `~/.ZBot/workspace` | 工作区目录 |
| `max_tokens` | `4396` | 最大输出 token 数 |
| `temperature` | `0.1` | 采样温度 |
| `memory_window` | `25` | 记忆窗口大小 |
| `max_tool_iterations` | `50` | 工具调用最大次数 |
| `tools.restrict_to_workspace` | `false` | 是否限制工作区 |

## 🔧 进阶配置

### 代理配置

```json
{
  "tools": {
    "web": {
      "proxy": "http://127.0.0.1:7890"
    }
  }
}
```

### 网页搜索配置

```json
{
  "tools": {
    "web": {
      "search": {
        "provider": "bocha",
        "api_key": "your_bocha_api_key",
        "max_results": 5
      }
    }
  }
}
```

### Shell 执行配置

```json
{
  "tools": {
    "exec": {
      "timeout": 60,
      "path_append": "/usr/local/bin"
    }
  }
}
```

## 📝 技术亮点

1. **多提供商集成**：通过 LiteLLM 统一接入 10+ LLM 提供商
2. **工具调用系统**：灵活的 Tool Registry 设计
3. **Agent 架构**：基于消息循环的智能体实现
4. **MCP 协议**：原生支持 Model Context Protocol
5. **跨平台兼容**：Windows/macOS/Linux 全平台支持
6. **安全控制**：工作区隔离、超时保护

## 🤝 贡献指南

欢迎贡献代码！请提交 Pull Request 或 Issue。

## 📄 许可证

MIT License

## 📮 联系方式

如有问题，请提交 GitHub Issue。