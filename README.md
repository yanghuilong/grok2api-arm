# Grok API Gateway

## 与原版差异

本 fork 版本相较于原版增加了以下功能：

0. **基本全部重写了...**
1. **自动获取 x-statsig-id** - 使用 Playwright 自动获取并管理认证头
2. **流模式标签过滤** - 自动移除响应中的 `<xaiArtifact` 等标签
3. **增强统计功能** - 改进的令牌使用统计和监控
4. **Grok4支持** - 反正我能用.jpg

## 环境变量配置

### 必需配置

| 环境变量 | 描述 | 默认值 | 示例 |
|---------|------|--------|------|
| `API_KEY` | API 访问密钥 | `sk-123456` | `sk-your-api-key` |
| `SSO` | Grok SSO 令牌（普通） | - | `token1,token2,token3` |
| `SSO_SUPER` | Grok SSO 令牌（超级） | - | `super_token1,super_token2` |

### 可选配置

| 环境变量 | 描述 | 默认值 | 有效值 | 示例 |
|---------|------|--------|--------|------|
| `IS_CUSTOM_SSO` | 允许动态 SSO 令牌 | `false` | `true/false` | `true` |
| `IS_TEMP_CONVERSATION` | 临时对话模式 | `true` | `true/false` | `false` |
| `SHOW_THINKING` | 显示推理过程 | `false` | `true/false` | `true` |
| `SHOW_SEARCH_RESULTS` | 显示搜索结果 | `true` | `true/false` | `false` |
| `IS_SUPER_GROK` | 启用超级 Grok 功能 | `false` | `true/false` | `true` |
| `MANAGER_SWITCH` | 启用 Web 管理界面 | - | `true/false` | `true` |
| `ADMINPASSWORD` | 管理界面密码 | - | 任意字符串 | `admin123` |
| `PORT` | 服务端口 | `5200` | 数字 | `8080` |
| `PROXY` | 代理服务器 | - | HTTP/SOCKS5 URL | `http://127.0.0.1:1080` |
| `CF_CLEARANCE` | Cloudflare 令牌 | - | CF 令牌字符串 | `cf_clearance_token` |
| `PICGO_KEY` | PICGO 图床 API 密钥 | - | 字符串 | `picgo_api_key` |
| `TUMY_KEY` | Tumy 图床 API 密钥 | - | 字符串 | `tumy_api_key` |
| `FILTERED_TAGS` | 过滤标签列表 | `xaiArtifact` | 逗号分隔 | `tag1,tag2,tag3` |

## 快速开始

### 使用 Docker Hub 镜像

现在可以直接从 Docker Hub 拉取预构建的镜像：

```bash
# 拉取镜像
docker pull verofess/grok2api

# 运行容器
docker run -d \
  --name grok2api \
  -p 5200:5200 \
  -e API_KEY=sk-your-api-key \
  -e SSO=your-sso-token \
  verofess/grok2api

# 或者使用 docker-compose
docker-compose up -d
```

### Docker Compose 示例

```yaml
services:
  grok2api:
    image: verofess/grok2api
    container_name: grok2api
    ports:
      - "5200:5200"
    environment:
      - API_KEY=sk-your-api-key
      - SSO=your-sso-token
      - IS_TEMP_CONVERSATION=true
      - SHOW_THINKING=false
    restart: unless-stopped
```