import os 
from langchain_huggingface import HuggingFaceEndpoint
import json
import pandas as pd
from langchain.tools import Tool

HUGGINGFACEHUB_API_TOKEN = "your-access-token"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN


llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.1,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)




# Load inventory CSV
def load_inventory(csv_path="inventory.csv"):
    return pd.read_csv(csv_path)

# Save updated inventory
def save_inventory(df, csv_path="inventory.csv"):
    df.to_csv(csv_path, index=False)

# Function to extract product name and quantity from user query using LLM
def parse_query(query, llm, product_list):
    prompt = (
        "Extract the product name and quantity from the following user request using the given product list. "
        "Return JSON only, at no cost should there eb any text after or other than the json.\n\n"
        "Product List: " + ", ".join(product_list) + "\n\n"
        "Example input:\n"
        "User: I want to order 2 wireless chargers.\n"
        "Output:\n"
        "{\"product_name\": \"wireless charger\", \"order_qty\": 2 }\n\n"
        "Now process this input:\n"
        f"User: {query}\n"
        "Output:"
    )
    
    response = llm.invoke(prompt).strip()
    response = response.split("\n")[-1]
    print("0"+response+"0")
    try:
        parsed_response = json.loads(response)
        return parsed_response.get("product_name"), parsed_response.get("order_qty")
    except json.JSONDecodeError:
        return None, None

# Function to check inventory
def check_inventory(product_name: str, order_qty: int):
    df = load_inventory()
    product_name = product_name.lower()
    
    if product_name not in df["Product Name"].str.lower().values:
        return f"Sorry, we do not have {product_name} in our inventory."
    
    product_row = df[df["Product Name"].str.lower() == product_name].iloc[0]
    available_qty = product_row["Quantity In Hand"]
    
    if available_qty <= 0:
        return f"{product_name} is out of stock. Would you like to check similar products?"
    elif available_qty < order_qty:
        return f"We only have {available_qty} of {product_name} in stock. Would you like to adjust your order?"
    
    # Update inventory
    df.loc[df["Product Name"].str.lower() == product_name, "Quantity In Hand"] -= order_qty
    df.loc[df["Product Name"].str.lower() == product_name, "Quantity Sold"] += order_qty
    save_inventory(df)
    return f"Order confirmed! Your {order_qty}x {product_name} will be processed."

# Define function for tool execution
def check_inventory_tool(input):
    return check_inventory(input["product_name"], input["order_qty"])

# Create react_agent with manual tool execution
def agent(user_query, llm=llm):
    df = load_inventory()
    product_list = df["Product Name"].str.lower().tolist()
    product_name, order_qty = parse_query(user_query, llm, product_list)
    if not product_name or not order_qty:
        return "Sorry, I couldn't understand your order. Could you please specify the product and quantity?"
    return check_inventory_tool({"product_name": product_name, "order_qty": order_qty})

proompt = """with the line given below give an input that is in human language. Do not give anything extra, all ypu have to do/
            is generate one simple sentence that conveys the line below in human language. Do not make up stuff.
            line: {line}

"""
# query = "I want to order 1 wireless keyboard"
# response = agent(query, llm)
# r = llm.invoke(proompt.format(line=response))
# print(r.split('\n')[-1])
