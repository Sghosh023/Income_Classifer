from pydantic import BaseModel

class IncomeClassifier(BaseModel):
    age: int
    occupation: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    workclass: str
    education: str
    marital_status: str
    race: str
    native_country: str
