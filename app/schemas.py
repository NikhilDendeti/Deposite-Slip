from pydantic import BaseModel, EmailStr, ConfigDict
from typing import Optional, List
from datetime import date, datetime

class ProcessingDetails(BaseModel):
    mode_used: str
    amount_source: Optional[str] = None  # "ocr", "rules", "llm", "manual"
    date_source: Optional[str] = None
    bank_source: Optional[str] = None
    account_source: Optional[str] = None
    ocr_confidence: Optional[float] = None
    rules_confidence: Optional[float] = None
    llm_confidence: Optional[float] = None

class UserCreate(BaseModel):
    email: EmailStr
    name: str
    password: str
    role: str = "accountant"
    branch_id: Optional[int] = None

class UserResponse(BaseModel):
    id: int
    email: str
    name: str
    role: str
    branch_id: Optional[int]
    is_active: bool

    model_config = ConfigDict(from_attributes=True)

class CollectionCreate(BaseModel):
    amount: float
    date: date
    branch_id: int
    description: Optional[str] = None

class CollectionResponse(BaseModel):
    id: int
    amount: float
    date: date
    branch_id: int
    description: Optional[str]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class DepositSlipCreate(BaseModel):
    collection_id: int
    manual_amount: Optional[float] = None
    manual_date: Optional[date] = None

class DepositSlipResponse(BaseModel):
    id: int
    collection_id: int
    file_path: str
    file_hash: Optional[str] = None
    ocr_amount: Optional[float]
    ocr_date: Optional[date]
    manual_amount: Optional[float]
    manual_date: Optional[date]
    bank_name: Optional[str]
    account_number: Optional[str]
    is_validated: bool
    validation_errors: Optional[str]
    confidence_score: float
    status: str
    override_reason: Optional[str] = None
    override_approved_by: Optional[str] = None
    processing_details: Optional[ProcessingDetails] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


