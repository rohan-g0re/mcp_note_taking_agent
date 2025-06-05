# 🤖 MCP Notetaking Agent

## What is MCP (Model Context Protocol)?

Think of MCP as a **universal translator** between AI applications and external tools/data sources.

**Before MCP:** Every AI app had to build custom integrations for each tool
- Claude needed custom code for GitHub
- Cursor needed custom code for databases  
- Each integration was unique = lots of duplicated work

**With MCP:** One standard protocol connects everything
- Build an MCP server once → works with ALL MCP clients
- Claude, Cursor, any MCP-compatible app can use it instantly

## Our Project Architecture

```
[User] → [MCP Client] → [Our MCP Server] → [Google Drive]
↓ ↓ ↓ ↓
Question → Protocol → Process & Save → Markdown File
```

## MCP Key Concepts

1. **MCP Server** (what we're building): Provides capabilities to AI clients
2. **MCP Client** (Claude, Cursor, etc.): Uses our server's capabilities  
3. **Tools**: Functions the AI can call (ask_question, save_note)
4. **Resources**: Data the AI can access (our saved notes)
5. **Prompts**: Template interactions for common tasks

## Setup

1. Create virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate it:
   ```bash
   # Windows
   venv\Scripts\activate
   
   # Mac/Linux  
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Current Phase: Foundation
- ✅ Project structure
- ✅ Dependencies installed
- 🔄 Understanding MCP basics
- ⏳ Building first components