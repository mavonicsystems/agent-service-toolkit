import requests
import os
from langchain_core.tools import BaseTool, tool

async def send_markdown_to_api_func(content: str, title: str, description: str, read_time: int, category: str) -> str:
    """Tool for publishing blog posts to the Dokimi API.
    
    Use this tool when you need to publish or save a completed blog post. The tool will send the 
    markdown content to the Dokimi blogging platform's API endpoint.
    
    Args:
        content: The markdown content of the blog post
        title: The title for the blog post
        description: A brief description of the blog post
        read_time: Estimated reading time in minutes
        category: The category name for the blog post
        
    Returns:
        Response from the API containing the published blog post details
    """
    url = "https://dokimi.ai/api/blog"
    headers = {
        "Content-Type": "application/json",
        "x-dokimi-api-key": os.getenv("DOKIMI_API_KEY")
    }
    data = {
        "content": content,
        "title": title,
        "description": description,
        "read_time": read_time,
        "category": category
    }
    
    response = requests.post(url, headers=headers, json=data)
    return response.json()

send_markdown_to_api: BaseTool = tool(send_markdown_to_api_func)
send_markdown_to_api.name = "SendMarkdownToAPI"