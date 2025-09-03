from sqlalchemy import Column, Integer, String, Float, Date, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
from passlib.context import CryptContext

Base = declarative_base()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="accountant")
    branch_id = Column(Integer, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())

    def set_password(self, password: str):
        self.hashed_password = pwd_context.hash(password)

    def verify_password(self, password: str) -> bool:
        return pwd_context.verify(password, self.hashed_password)

class Collection(Base):
    __tablename__ = "collections"

    id = Column(Integer, primary_key=True, index=True)
    amount = Column(Float, nullable=False)
    date = Column(Date, nullable=False)
    branch_id = Column(Integer, nullable=False)
    collected_by = Column(Integer, ForeignKey("users.id"))
    description = Column(Text)
    created_at = Column(DateTime, server_default=func.now())

    collector = relationship("User", foreign_keys=[collected_by])
    deposit_slips = relationship("DepositSlip", back_populates="collection")

class DepositSlip(Base):
    __tablename__ = "deposit_slips"

    id = Column(Integer, primary_key=True, index=True)
    collection_id = Column(Integer, ForeignKey("collections.id"), nullable=False)
    file_path = Column(String, nullable=False)
    file_hash = Column(String, nullable=True, index=True)

    ocr_amount = Column(Float)
    ocr_date = Column(Date)
    bank_name = Column(String)
    account_number = Column(String)

    manual_amount = Column(Float)
    manual_date = Column(Date)

    is_validated = Column(Boolean, default=False)
    validation_errors = Column(Text)
    confidence_score = Column(Float, default=0.0)
    status = Column(String, default="pending")

    override_reason = Column(Text)
    override_approved_by = Column(String)
    override_at = Column(DateTime)
    processing_details = Column(Text)  # JSON string

    processed_by = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    collection = relationship("Collection", back_populates="deposit_slips")
    processor = relationship("User", foreign_keys=[processed_by])

