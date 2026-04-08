# Nanobot MCP 服务器使用指南

> 适合零基础小白的详细教程

---

## 目录

1. [什么是 MCP？](#一什么是-mcp)
2. [准备工作](#二准备工作)
3. [配置文件位置](#三配置文件位置)
4. [三种连接方式详解](#四三种连接方式详解)
5. [手把手配置示例](#五手把手配置示例)
6. [常用 MCP 服务器推荐](#六常用-mcp-服务器推荐)
7. [工具命名规则](#七工具命名规则)
8. [常见问题排查](#八常见问题排查)
9. [进阶配置](#九进阶配置)

---

## 一、什么是 MCP？

### 1.1 简单理解

**MCP (Model Context Protocol)** 是一个标准化协议，让 AI 模型能够通过统一接口访问外部工具和服务。

**打个比方**：
- Nanobot 就像是一个"智能管家"
- MCP 服务器就像是"工具箱"
- 通过 MCP 协议，管家可以打开工具箱，使用里面的各种工具

### 1.2 能做什么？

通过 MCP，你的 AI 助手可以：

- **操作文件系统**：读取、写入、管理文件
- **搜索网络**：使用 Brave 搜索、Google 搜索等
- **连接数据库**：查询和操作数据库
- **操作 GitHub**：创建 Issue、PR、查看代码
- **浏览器自动化**：打开网页、截图、提取内容
- **持久化记忆**：让 AI 记住之前的对话

### 1.3 工作原理

```
┌─────────────┐     MCP 协议     ┌─────────────┐
│   Nanobot   │ ◄──────────────► │ MCP 服务器   │
│   (AI助手)   │                  │  (工具箱)    │
└─────────────┘                  └─────────────┘
      │                                │
      │  1. AI 决定使用工具             │
      │  2. 通过 MCP 协议发送请求        │
      │  3. MCP 服务器执行操作          │
      │  4. 返回结果给 AI               │
      ▼                                ▼
   用户看到结果                   实际执行操作
```

---

## 二、准备工作

### 2.1 安装 Node.js（stdio 模式需要）

大多数 MCP 服务器使用 Node.js 运行，需要先安装：

**Windows 安装步骤**：

1. 打开浏览器，访问 https://nodejs.org/
2. 下载 LTS（长期支持版）安装包
3. 双击安装包，一路点击"下一步"完成安装
4. 打开命令行，验证安装：
   ```bash
   node -v
   npm -v
   ```
   如果显示版本号，说明安装成功

### 2.2 安装 Python MCP 依赖

Nanobot 需要 `mcp` 包来连接 MCP 服务器：

```bash
pip install mcp
```

### 2.3 验证准备完成

```bash
# 检查 Node.js
node -v

# 检查 npm
npm -v

# 检查 Python MCP 包
python -c "import mcp; print('MCP 包已安装')"
```

---

## 三、配置文件位置

### 3.1 配置文件路径

Nanobot 的配置文件位于：

```
C:\Users\你的用户名\.nanobot\config.json
```

例如，如果你的用户名是 `15927`，则路径是：

```
C:\Users\15927\.nanobot\config.json
```

### 3.2 打开配置文件

**方法一：使用 VS Code 打开**

1. 打开 VS Code
2. 按 `Ctrl + O` 打开文件
3. 在地址栏输入 `C:\Users\15927\.nanobot\config.json`
4. 回车打开

**方法二：使用命令行打开**

```bash
notepad C:\Users\15927\.nanobot\config.json
```

### 3.3 配置文件结构

配置文件是 JSON 格式，MCP 配置在 `tools.mcpServers` 字段：

```json
{
  "workspace": "~/.nanobot/workspace",
  "model": "Qwen/Qwen2.5-72B-Instruct",
  "provider": "siliconflow",
  "tools": {
    "mcpServers": {
      "这里填写 MCP 服务器配置"
    }
  }
}
```

---

## 四、三种连接方式详解

MCP 支持三种连接方式，根据你的需求选择：

### 4.1 stdio 模式（本地进程）⭐ 推荐

**适用场景**：运行本地安装的 MCP 服务器程序

**工作原理**：
```
Nanobot 启动子进程 ←→ 通过标准输入输出通信 ←→ MCP 服务器程序
```

**配置字段说明**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `command` | 字符串 | 是 | 要执行的命令，如 `npx`、`python` |
| `args` | 列表 | 是 | 命令参数列表 |
| `env` | 字典 | 否 | 环境变量 |
| `enabled_tools` | 列表 | 否 | 启用哪些工具，`["*"]` 表示全部 |
| `tool_timeout` | 数字 | 否 | 工具调用超时时间（秒），默认 30 |

**配置示例**：

```json
{
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:\\Users\\15927\\Documents"],
  "env": {},
  "enabled_tools": ["*"],
  "tool_timeout": 30
}
```

### 4.2 SSE 模式（服务器推送事件）

**适用场景**：连接支持 Server-Sent Events 的远程服务器

**工作原理**：
```
Nanobot ←── HTTP 长连接 ──→ 远程 MCP 服务器
         （服务器主动推送消息）
```

**配置字段说明**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `url` | 字符串 | 是 | 服务器地址，通常以 `/sse` 结尾 |
| `headers` | 字典 | 否 | HTTP 请求头，用于认证 |

**配置示例**：

```json
{
  "url": "https://api.example.com/mcp/sse",
  "headers": {
    "Authorization": "Bearer your-api-key"
  }
}
```

### 4.3 streamableHttp 模式（流式 HTTP）

**适用场景**：现代 HTTP 流式 MCP 服务

**工作原理**：
```
Nanobot ←── HTTP 流式连接 ──→ 远程 MCP 服务器
         （双向流式通信）
```

**配置字段说明**：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `url` | 字符串 | 是 | 服务器地址 |
| `headers` | 字典 | 否 | HTTP 请求头 |

**配置示例**：

```json
{
  "url": "https://api.example.com/mcp",
  "headers": {
    "Authorization": "Bearer your-api-key"
  }
}
```

---

## 五、手把手配置示例

### 5.1 示例一：配置文件系统 MCP 服务器

**目标**：让 AI 能够操作你电脑上的文件

**步骤**：

1. **打开配置文件**

   ```bash
   notepad C:\Users\15927\.nanobot\config.json
   ```

2. **找到 `mcpServers` 字段**，修改为：

   ```json
   {
     "workspace": "~/.nanobot/workspace",
     "model": "Pro/MiniMaxAI/MiniMax-M2.5",
     "provider": "siliconflow",
     "tools": {
       "mcpServers": {
         "fs": {
           "command": "npx",
           "args": [
             "-y",
             "@modelcontextprotocol/server-filesystem",
             "C:\\Users\\15927\\Documents"
           ],
           "enabled_tools": ["*"],
           "tool_timeout": 30
         }
       }
     }
   }
   ```

   **注意**：把 `C:\\Users\\15927\\Documents` 改成你允许 AI 访问的目录路径。

3. **保存配置文件**（`Ctrl + S`）

4. **重启 Nanobot**

   ```bash
   python -m nanobot agent
   ```

5. **测试使用**

   在对话中输入：
   ```
   列出我的文档目录下有哪些文件
   ```

   AI 会自动调用 `mcp_fs_list_directory` 工具。

### 5.2 示例二：配置 GitHub MCP 服务器

**目标**：让 AI 能够操作 GitHub（创建 Issue、PR 等）

**步骤**：

1. **获取 GitHub Token**

   - 访问 https://github.com/settings/tokens
   - 点击 "Generate new token (classic)"
   - 勾选需要的权限（如 `repo`、`issues`）
   - 点击 "Generate token"
   - 复制生成的 token（以 `ghp_` 开头）

2. **修改配置文件**

   ```json
   {
     "tools": {
       "mcpServers": {
         "github": {
           "command": "npx",
           "args": ["-y", "@modelcontextprotocol/server-github"],
           "env": {
             "GITHUB_TOKEN": "ghp_你的token"
           },
           "enabled_tools": ["*"],
           "tool_timeout": 60
         }
       }
     }
   }
   ```

3. **重启 Nanobot**

4. **测试使用**

   ```
   帮我查看 anthropics/claude-code 这个仓库最近的 issues
   ```

### 5.3 示例三：配置记忆 MCP 服务器

**目标**：让 AI 拥有持久化记忆能力

**步骤**：

1. **修改配置文件**

   ```json
   {
     "tools": {
       "mcpServers": {
         "memory": {
           "command": "npx",
           "args": ["-y", "@modelcontextprotocol/server-memory"],
           "enabled_tools": ["*"],
           "tool_timeout": 30
         }
       }
     }
   }
   ```

2. **重启 Nanobot**

3. **测试使用**

   ```
   记住：我最喜欢的编程语言是 Python

   （稍后问它）
   我最喜欢的编程语言是什么？
   ```

### 5.4 示例四：同时配置多个 MCP 服务器

```json
{
  "workspace": "~/.nanobot/workspace",
  "model": "Qwen/Qwen2.5-72B-Instruct",
  "provider": "siliconflow",
  "tools": {
    "mcpServers": {
      "fs": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:\\Users\\15927\\Documents"]
      },
      "memory": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-memory"]
      },
      "github": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-github"],
        "env": {
          "GITHUB_TOKEN": "ghp_你的token"
        }
      }
    }
  }
}
```

---

## 六、常用 MCP 服务器推荐

### 6.1 官方 MCP 服务器

| 服务器名称 | 安装命令 | 功能说明 | 需要的配置 |
|-----------|---------|---------|-----------|
| filesystem | `@modelcontextprotocol/server-filesystem` | 文件系统操作 | 指定允许访问的目录 |
| memory | `@modelcontextprotocol/server-memory` | 持久化记忆 | 无 |
| github | `@modelcontextprotocol/server-github` | GitHub 操作 | GitHub Token |
| brave-search | `@modelcontextprotocol/server-brave-search` | Brave 搜索 | Brave API Key |
| puppeteer | `modelcontextprotocol/server-puppeteer` | 浏览器自动化 | 无 |
| slack | `@modelcontextprotocol/server-slack` | Slack 操作 | Slack Token |
| google-maps | `@modelcontextprotocol/server-google-maps` | Google 地图 | Google API Key |

### 6.2 社区 MCP 服务器

可以在以下地方找到更多 MCP 服务器：

- GitHub 搜索：https://github.com/search?q=mcp+server
- MCP 官方列表：https://github.com/modelcontextprotocol/servers

---

## 七、工具命名规则

### 7.1 命名格式

MCP 工具注册后，名称格式为：

```
mcp_{服务器名}_{原始工具名}
```

### 7.2 示例

假设配置了一个名为 `fs` 的文件系统 MCP 服务器：

| 原始工具名 | 注册后名称 | 功能 |
|-----------|-----------|------|
| `read_file` | `mcp_fs_read_file` | 读取文件 |
| `write_file` | `mcp_fs_write_file` | 写入文件 |
| `list_directory` | `mcp_fs_list_directory` | 列出目录 |
| `create_directory` | `mcp_fs_create_directory` | 创建目录 |
| `delete_file` | `mcp_fs_delete_file` | 删除文件 |

### 7.3 为什么这样命名？

避免不同服务器的工具重名冲突。

例如，你同时配置了两个服务器：
- `fs1`：访问目录 A
- `fs2`：访问目录 B

它们的工具分别是：
- `mcp_fs1_read_file`
- `mcp_fs2_read_file`

AI 可以区分调用哪个服务器的工具。

---

## 八、常见问题排查

### 8.1 问题：MCP 服务器连接失败

**错误信息**：
```
MCP 服务器 'xxx' 连接失败: ...
```

**排查步骤**：

1. **检查 Node.js 是否安装**

   ```bash
   node -v
   npm -v
   ```

   如果没有显示版本号，请安装 Node.js。

2. **检查网络连接**

   ```bash
   ping registry.npmjs.org
   ```

   npx 需要从 npm 下载包，确保网络畅通。

3. **手动测试 MCP 服务器**

   ```bash
   npx -y @modelcontextprotocol/server-filesystem C:\test
   ```

   如果报错，说明服务器包有问题。

### 8.2 问题：工具未注册

**错误信息**：
```
MCP 服务器 'xxx' 中，enabledTools 指定的这些工具未找到: ...
```

**原因**：`enabled_tools` 中指定的工具名称不正确

**解决方案**：

1. 使用 `["*"]` 启用所有工具
2. 查看日志，找到正确的工具名称
3. 使用**原始工具名**，不是包装后的名称

```json
{
  "enabled_tools": ["read_file"]  // 正确
}
```

```json
{
  "enabled_tools": ["mcp_fs_read_file"]  // 错误！应该用原始名称
}
```

### 8.3 问题：工具调用超时

**错误信息**：
```
MCP 工具 'xxx' 调用超时（30 秒）
```

**解决方案**：增大超时时间

```json
{
  "tool_timeout": 120
}
```

### 8.4 问题：认证失败

**错误信息**：
```
认证失败 / Unauthorized
```

**排查步骤**：

1. 检查 API Key 是否正确
2. 检查环境变量名称是否正确（如 `GITHUB_TOKEN`）
3. 检查 Token 是否有过期时间

### 8.5 问题：Windows 路径问题

**错误**：路径中的反斜杠被转义

**错误写法**：
```json
{
  "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:\Users\15927\Documents"]
}
```

**正确写法**（双反斜杠或正斜杠）：
```json
{
  "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:\\Users\\15927\\Documents"]
}
```

或

```json
{
  "args": ["-y", "@modelcontextprotocol/server-filesystem", "C:/Users/15927/Documents"]
}
```

---

## 九、进阶配置

### 9.1 环境变量配置

某些 MCP 服务器需要 API Key，可以通过环境变量配置：

```json
{
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-github"],
  "env": {
    "GITHUB_TOKEN": "ghp_xxxxx",
    "ANOTHER_VAR": "value"
  }
}
```

### 9.2 工具过滤

只启用特定工具，减少 AI 可用的工具数量：

```json
{
  "enabled_tools": ["read_file", "list_directory"]
}
```

### 9.3 多目录访问（filesystem）

文件系统 MCP 支持配置多个可访问目录：

```json
{
  "command": "npx",
  "args": [
    "-y",
    "@modelcontextprotocol/server-filesystem",
    "C:\\Users\\15927\\Documents",
    "C:\\Users\\15927\\Downloads",
    "D:\\Projects"
  ]
}
```

### 9.4 自定义超时配置

不同服务器设置不同超时时间：

```json
{
  "mcpServers": {
    "fast-server": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"],
      "tool_timeout": 10
    },
    "slow-server": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-puppeteer"],
      "tool_timeout": 120
    }
  }
}
```

---

## 十、快速检查清单

在配置 MCP 服务器前，请确认：

- [ ] 已安装 Node.js（stdio 模式需要）
- [ ] 已安装 Python mcp 包（`pip install mcp`）
- [ ] 配置文件路径正确
- [ ] JSON 格式正确（可以用 JSON 校验工具检查）
- [ ] Windows 路径使用双反斜杠或正斜杠
- [ ] 需要认证的服务器已配置 Token

---

## 十一、获取帮助

如果遇到问题：

1. 查看 Nanobot 日志输出
2. 检查 MCP 服务器文档
3. 在 GitHub 上搜索相关问题
4. 提交 Issue 到 Nanobot 仓库

---

**祝你使用愉快！**
