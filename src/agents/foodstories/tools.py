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

class StoreInfo(BaseModel):
    id: int = Field(
        ...,
        description="Store ID",
        example=1
    )
    code: str = Field(
        ...,
        description="Store code",
        example="vasantkunj"
    )
    name: str = Field(
        ...,
        description="Store name",
        example="Vasant Kunj, New Delhi"
    )
    website_id: int = Field(
        ...,
        description="Associated website ID",
        example=1
    )
    store_group_id: int = Field(
        ...,
        description="Store group ID",
        example=1
    )
    is_active: int = Field(
        ...,
        description="Store active status (1 for active)",
        example=1
    )

async def get_customer_info_func(mobile_number: str) -> Dict[str, Any]:
    """Tool for retrieving customer information from the FoodStories API.
    
    Use this tool to get detailed customer information by searching with mobile number.
    
    Args:
        mobile_number: Customer's mobile number to search for
        
    Returns:
        Customer information including personal details and addresses
    """
    base_url = os.getenv('FOODSTORIES_API_URL', 'https://uat.api.foodstories.shop')
    url = f"{base_url}/rest/V1/customers/search"
    params = {
        "searchCriteria[filterGroups][0][filters][0][field]": "mobile_number",
        "searchCriteria[filterGroups][0][filters][0][value]": mobile_number,
        "searchCriteria[filterGroups][0][filters][0][conditionType]": "eq"
    }
    headers = {
        "Authorization": f"Bearer {os.getenv('FOODSTORIES_API_KEY')}"
    }

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()

get_customer_info: BaseTool = tool(get_customer_info_func)
get_customer_info.name = "GetCustomerInfo"

async def list_orders_func(customer_id: str, page_size: int = 5) -> List[Dict[str, Any]]:
    """Tool for retrieving a customer's order history from FoodStories.
    
    Use this tool to get a list of orders placed by the specified customer, including
    order details like ID, total amount, and current status.
    
    Args:
        customer_id: The unique identifier for the customer
        page_size: Number of orders to return per page (default: 5)
        
    Returns:
        List of order details including order ID, total, and status
    """
    base_url = os.getenv('FOODSTORIES_API_URL', 'https://uat.api.foodstories.shop')
    url = f"{base_url}/rest/V1/orders"
    params = {
        "searchCriteria[filterGroups][0][filters][0][field]": "customer_id",
        "searchCriteria[filterGroups][0][filters][0][value]": customer_id,
        "searchCriteria[filterGroups][0][filters][0][conditionType]": "eq",
        "searchCriteria[pageSize]": page_size,
        "searchCriteria[sortOrders][0][direction]": "desc",
        "searchCriteria[sortOrders][0][field]": "entity_id"
    }
    headers = {
        "Authorization": f"Bearer {os.getenv('FOODSTORIES_API_KEY')}"
    }

    response = requests.get(url, headers=headers, params=params)
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
    base_url = os.getenv('FOODSTORIES_API_URL', 'https://uat.api.foodstories.shop')
    url = f"{base_url}/rest/V1/orders"
    params = {
        "searchCriteria[filterGroups][0][filters][0][field]": "entity_id",
        "searchCriteria[filterGroups][0][filters][0][value]": order_id,
        "searchCriteria[filterGroups][0][filters][0][conditionType]": "eq"
    }
    headers = {
        "Authorization": f"Bearer {os.getenv('FOODSTORIES_API_KEY')}"
    }

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()

get_order_details: BaseTool = tool(get_order_details_func)
get_order_details.name = "GetOrderDetails"

async def list_stores_func() -> List[Dict[str, Any]]:
    """Tool for retrieving the list of stores from FoodStories.
    
    Use this tool to get a list of all available stores, including their IDs, codes,
    names, and active status. The stores are sorted by ID in descending order.
    
    Returns:
        List of store information including ID, code, name, and status
    """
    base_url = os.getenv('FOODSTORIES_API_URL', 'https://uat.api.foodstories.shop')
    url = f"{base_url}/rest/V1/store/storeViews"
    headers = {
        "Authorization": f"Bearer {os.getenv('FOODSTORIES_API_KEY')}"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

list_stores: BaseTool = tool(list_stores_func)
list_stores.name = "ListStores"
