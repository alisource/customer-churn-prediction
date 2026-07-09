from pydantic import BaseModel, Field

class ChurnDataSchema(BaseModel):
    customer_id: int
    age: int
    gender: str
    city: str
    tenure_months : int
    avg_order_value : int
    total_orders : int
    last_purchase_days_ago : int
    support_tickets : int
    subscription_type : str
    churn : int
      # 1 jika churn, 0 jika tidak