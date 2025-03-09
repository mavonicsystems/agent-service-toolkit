import requests
import os
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

class MetaSEO(BaseModel):
    keywords: Optional[List[str]] = Field(
        None, 
        description="List of SEO keywords",
        example=["python", "programming", "web development", "tutorial"]
    )
    jsonLD: Optional[Dict] = Field(
        None, 
        description="JSON-LD structured data",
        example={
            "@type": "BlogPosting",
            "headline": "Getting Started with Python",
            "author": {"@type": "Person", "name": "John Doe"},
            "datePublished": "2024-01-20"
        }
    )
    canonical_url: Optional[str] = Field(
        None, 
        description="Canonical URL for the post",
        example="https://dokimi.ai/blog/getting-started-with-python"
    )
    meta_title: Optional[str] = Field(
        None, 
        description="SEO optimized title",
        example="Getting Started with Python: A Complete Beginner's Guide | Dokimi"
    )
    meta_description: Optional[str] = Field(
        None, 
        description="SEO optimized description",
        example="Learn Python programming from scratch with this comprehensive beginner's guide. Covers basic syntax, data types, and practical examples."
    )
    og_image: Optional[str] = Field(
        None, 
        description="Open Graph image URL",
        example="https://dokimi.ai/images/blog/python-beginners-guide.jpg"
    )
    twitter_card: Optional[Dict] = Field(
        None, 
        description="Twitter card metadata",
        example={
            "card": "summary_large_image",
            "site": "@DokimiAI",
            "title": "Getting Started with Python",
            "description": "Learn Python programming from scratch"
        }
    )

class BlogPostInput(BaseModel):
    content: str = Field(
        ..., 
        description="The markdown content of the blog post",
        example="# Getting Started with Python\n\nPython is a versatile programming language..."
    )
    title: str = Field(
        ..., 
        description="The title for the blog post",
        example="Getting Started with Python: A Beginner's Guide"
    )
    description: str = Field(
        ..., 
        description="A brief description of the blog post",
        example="A comprehensive guide to help beginners start their Python programming journey"
    )
    read_time: int = Field(
        ..., 
        description="Estimated reading time in minutes",
        example=5
    )
    category: str = Field(
        ..., 
        description="The category name for the blog post",
        example="Programming Tutorials"
    )
    meta_seo: Optional[MetaSEO] = Field(
        None, 
        description="SEO metadata for the blog post"
    )

async def send_markdown_to_api_func(blog_post: BlogPostInput) -> str:
    """Tool for publishing blog posts to the Dokimi API.
    
    Use this tool when you need to publish or save a completed blog post. The tool will send the 
    markdown content to the Dokimi blogging platform's API endpoint.
    
    Args:
        blog_post: BlogPostInput containing all the required fields for publishing a blog post
        
    Returns:
        Response from the API containing the published blog post details
    """
    url = "https://dokimi.ai/api/blog"
    headers = {
        "Content-Type": "application/json",
        "x-dokimi-api-key": os.getenv("DOKIMI_API_KEY")
    }
    data = blog_post.model_dump(exclude_none=True)
    
    response = requests.post(url, headers=headers, json=data)
    return response.json()

send_markdown_to_api: BaseTool = tool(send_markdown_to_api_func)
send_markdown_to_api.name = "SendMarkdownToAPI"