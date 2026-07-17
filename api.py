from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
from backend import optimize_price  # This imports your ML function!

# Initialize the FastAPI server
app = FastAPI(title="Dynamic Pricing API")
# Allow web pages to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all websites to talk to your API
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the format of the data the website will send us
class PricingRequest(BaseModel):
    category_avg_price: float
    avg_freight: float
    is_weekend: int
    month: int

# Create the web endpoint (URL) that listens for data
@app.post("/optimize")
def get_optimized_price(data: PricingRequest):
    
    # Plug the website's data into your machine learning function
    optimal_p, max_rev = optimize_price(
        category_avg_price=data.category_avg_price,
        avg_freight=data.avg_freight,
        is_weekend=data.is_weekend,
        month=data.month
    )
    
    # FastAPI will automatically convert this Python dictionary into JSON for the web!
    return {
        "status": "success",
        "optimal_price": float(optimal_p),
        "max_revenue": float(max_rev)
    }