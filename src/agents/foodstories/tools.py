import os
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field
from typing import Dict, Any, List
import requests

class CustomerInfo(BaseModel):
    name: str = Field(
        ...,
        description="Customer's full name",
        example="John Smith"
    )
    email: str = Field(
        ...,
        description="Customer's email address", 
        example="john.smith@email.com"
    )
    addresses: List[Dict] = Field(
        ...,
        description="List of customer addresses",
        example=[{
            "street": "123 Main St",
            "city": "New York",
            "state": "NY",
            "zip": "10001"
        }]
    )
    subscription_status: str = Field(
        ...,
        description="Customer's subscription status",
        example="active"
    )

class OrderInfo(BaseModel):
    order_id: str = Field(
        ...,
        description="Unique order identifier",
        example="entity_id from the response of ListOrders tool"
    )
    total: float = Field(
        ...,
        description="Total order amount",
        example=99.99
    )
    status: str = Field(
        ...,
        description="Current order status",
        example="processing"
    )
    items: List[Dict] = Field(
        ...,
        description="List of items in the order",
        example=[{
            "name": "Product Name",
            "quantity": 2,
            "price": 49.99
        }]
    )

async def get_customer_info_func() -> Dict[str, Any]:
    """Tool for retrieving customer information from the FoodStories API.
    
    Use this tool to get detailed customer information including personal details,
    addresses, and subscription status.
    
    Returns:
        Customer information including personal details and addresses
    """
    url = "https://uat.api.foodstories.shop/rest/V1/customers/me"
    headers = {
        "Authorization": f"Bearer {os.getenv('FOODSTORIES_API_KEY')}"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

get_customer_info: BaseTool = tool(get_customer_info_func)
get_customer_info.name = "GetCustomerInfo"

async def list_orders_func() -> List[Dict[str, Any]]:
    """Tool for retrieving a customer's order history from FoodStories.
    
    Use this tool to get a list of all orders placed by the customer, including
    order details like ID, total amount, and current status.
    
    Returns:
        List of order details including order ID, total, and status
    """
    url = "https://uat.api.foodstories.shop/rest/V1/me/orders"
    headers = {
        "Authorization": f"Bearer {os.getenv('FOODSTORIES_API_KEY')}"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

list_orders: BaseTool = tool(list_orders_func)
list_orders.name = "ListOrders"


async def get_order_details_func(order_id: str) -> Dict[str, Any]:
    """Tool for retrieving detailed order information from FoodStories.
    
    Use this tool to get a specific order's details including items, total, and status.
    
    Args:
        order_id: The unique identifier for the order
        
    Returns:
        Detailed order information including items, total, and status
    """
    url = f"https://uat.api.foodstories.shop/rest/V1/me/orders/{order_id}"
    headers = {
        "Authorization": f"Bearer {os.getenv('FOODSTORIES_API_KEY')}"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

get_order_details: BaseTool = tool(get_order_details_func)
get_order_details.name = "GetOrderDetails"
