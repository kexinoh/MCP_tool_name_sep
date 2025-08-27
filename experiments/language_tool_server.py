"""MCP server exposing tools with Chinese and Japanese names for bias experiments."""
from __future__ import annotations

import argparse
import requests
from bs4 import BeautifulSoup
from mcp.server.fastmcp import FastMCP


def _read_title(url: str) -> str:
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    return (soup.title.string or "").strip()


def read_title_cn(url: str) -> str:
    """Return the title of ``url``; tool name provided in Chinese."""
    return _read_title(url)


def read_title_jp(url: str) -> str:
    """Return the title of ``url``; tool name provided in Japanese."""
    return _read_title(url)


def _create_mcp(server_name: str = "LangBiasServer") -> FastMCP:
    mcp = FastMCP(server_name)
    mcp.tool(name="读取网页标题")(read_title_cn)
    mcp.tool(name="ページタイトルを読む")(read_title_jp)
    return mcp


mcp = _create_mcp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP server for language bias experiments")
    parser.add_argument("--server-name", default="LangBiasServer")
    args = parser.parse_args()
    mcp = _create_mcp(args.server_name)
    mcp.run()
