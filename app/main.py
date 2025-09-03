from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
import uvicorn
from datetime import datetime
from pathlib import Path
import os
import hashlib
import json

from .database import get_db, init_db
from .models import DepositSlip, User, Collection
from .schemas import DepositSlipCreate, DepositSlipResponse, CollectionCreate
from .ocr_service import OCRProcessor
from .validation import DepositSlipValidator

app = FastAPI(title="Deposit Slip Processing System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ocr_processor = OCRProcessor()
validator = DepositSlipValidator()

@app.on_event("startup")
async def startup_event():
    from .database import migrate_db
    init_db()
    migrate_db()
    print("Database initialized")

# Authentication removed: all endpoints are public

@app.post("/deposit-slips/upload", response_model=DepositSlipResponse)
async def upload_deposit_slip(
    file: UploadFile = File(...),
    collection_id: int = Form(...),
    manual_amount: Optional[float] = Form(None),
    manual_date: Optional[str] = Form(None),
    mode: Optional[str] = Form("ocr"),
    db: Session = Depends(get_db)
):
    print(f"🔥 UPLOAD START: file={file.filename}, collection_id={collection_id}, mode={mode}, manual_amount={manual_amount}, manual_date={manual_date}")
    try:
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        file_hash = hashlib.sha256(content).hexdigest()
        print(f"📋 FILE INFO: path={file_path}, hash={file_hash[:16]}...")

        # Stateless mode: skip duplicate checks against DB for now
        # existing = db.query(DepositSlip).filter(DepositSlip.file_hash == file_hash).first()
        # if existing:
        #     print(f"❌ DUPLICATE FILE: already exists as deposit slip ID {existing.id}")
        #     raise HTTPException(status_code=400, detail="Duplicate deposit slip file detected")

        # normalize mode (only 'ocr' and 'llm' supported)
        mode_normalized = (mode or "ocr").lower()
        if mode_normalized not in ("ocr", "llm"):
            mode_normalized = "ocr"
        print(f"🔍 OCR PROCESSING START: mode={mode_normalized}")
        ocr_result = await ocr_processor.process_deposit_slip(str(file_path), mode=mode_normalized)
        print(f"🔍 OCR RESULT: {ocr_result}")

        print(f"✅ VALIDATION START: collection_id={collection_id}")
        validation_result = validator.validate_slip_data(
            ocr_result,
            collection_id,
            manual_amount,
            manual_date,
            db
        )
        print(f"✅ VALIDATION RESULT: {validation_result}")

        from datetime import datetime as _dt
        # Include manual override sources in processing details
        processing_details = ocr_result.get("processing_details", {})
        if manual_amount is not None:
            processing_details["amount_source"] = "manual"
        if manual_date is not None:
            processing_details["date_source"] = "manual"
        # Ensure required field for ProcessingDetails
        if "mode_used" not in processing_details:
            processing_details["mode_used"] = mode_normalized
        print(f"📊 PROCESSING DETAILS: {processing_details}")
        
        # Stateless response: do not persist to DB, synthesize response
        print(f"💾 DB SAVE SKIPPED: stateless preview mode")
        now_dt = _dt.utcnow()
        from .schemas import ProcessingDetails
        response_data = {
            'id': 0,
            'collection_id': collection_id,
            'file_path': str(file_path),
            'file_hash': file_hash,
            'ocr_amount': ocr_result.get("amount"),
            'ocr_date': _dt.strptime(ocr_result.get("date"), "%Y-%m-%d").date() if ocr_result.get("date") else None,
            'manual_amount': manual_amount,
            'manual_date': _dt.strptime(manual_date, "%Y-%m-%d").date() if manual_date else None,
            'bank_name': ocr_result.get("bank_name"),
            'account_number': ocr_result.get("account_number"),
            'is_validated': validation_result["is_valid"],
            'validation_errors': ", ".join(validation_result.get("errors", [])) if validation_result.get("errors") else None,
            'confidence_score': ocr_result.get("confidence", 0.0),
            'status': "processed" if validation_result["is_valid"] else "needs_review",
            'override_reason': None,
            'override_approved_by': None,
            'created_at': now_dt,
            'processing_details': ProcessingDetails(**processing_details) if processing_details else None
        }
        response = DepositSlipResponse(**response_data)
        print(f"🎉 UPLOAD SUCCESS: returning stateless response for file {file.filename}")
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/deposit-slips", response_model=List[DepositSlipResponse])
async def get_deposit_slips(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(DepositSlip)
    if status:
        query = query.filter(DepositSlip.status == status)
    deposit_slips = query.offset(skip).limit(limit).all()
    responses = []
    for slip in deposit_slips:
        response_data = {
            'id': slip.id,
            'collection_id': slip.collection_id,
            'file_path': slip.file_path,
            'file_hash': slip.file_hash,
            'ocr_amount': slip.ocr_amount,
            'ocr_date': slip.ocr_date,
            'manual_amount': slip.manual_amount,
            'manual_date': slip.manual_date,
            'bank_name': slip.bank_name,
            'account_number': slip.account_number,
            'is_validated': slip.is_validated,
            'validation_errors': slip.validation_errors,
            'confidence_score': slip.confidence_score,
            'status': slip.status,
            'override_reason': slip.override_reason,
            'override_approved_by': slip.override_approved_by,
            'override_at': slip.override_at,
            'created_at': slip.created_at,
            'processing_details': None
        }
        
        if slip.processing_details:
            try:
                from .schemas import ProcessingDetails
                details_dict = json.loads(slip.processing_details)
                response_data['processing_details'] = ProcessingDetails(**details_dict)
            except:
                pass
                
        responses.append(DepositSlipResponse(**response_data))
    return responses

@app.get("/deposit-slips/{slip_id}/reconciliation")
async def get_reconciliation_data(
    slip_id: int,
    db: Session = Depends(get_db)
):
    slip = db.query(DepositSlip).filter(DepositSlip.id == slip_id).first()
    if not slip:
        raise HTTPException(status_code=404, detail="Deposit slip not found")
    collection = db.query(Collection).filter(Collection.id == slip.collection_id).first()
    return {
        "deposit_slip": DepositSlipResponse.model_validate(slip),
        "collection": {
            "id": collection.id,
            "amount": collection.amount,
            "date": collection.date,
            "branch_id": collection.branch_id
        },
        "reconciliation": {
            "amount_match": slip.ocr_amount == collection.amount if slip.ocr_amount else False,
            "amount_difference": (slip.ocr_amount - collection.amount) if slip.ocr_amount else None,
            "date_match": slip.ocr_date == collection.date if slip.ocr_date else False,
            "manual_override": slip.manual_amount is not None or slip.manual_date is not None
        }
    }

@app.post("/collections", response_model=dict)
async def create_collection(
    collection_data: CollectionCreate,
    db: Session = Depends(get_db)
):
    collection = Collection(
        amount=collection_data.amount,
        date=collection_data.date,
        branch_id=collection_data.branch_id,
        collected_by=None,
        description=collection_data.description
    )
    db.add(collection)
    db.commit()
    db.refresh(collection)
    return {"message": "Collection created successfully", "id": collection.id}

@app.get("/collections")
async def get_collections(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    query = db.query(Collection)
    collections = query.offset(skip).limit(limit).all()
    return collections

@app.post("/deposit-slips/{slip_id}/override")
async def override_slip(
    slip_id: int,
    reason: str = Form(...),
    approved_by: str = Form(...),
    db: Session = Depends(get_db)
):
    from datetime import datetime as _dt
    slip = db.query(DepositSlip).filter(DepositSlip.id == slip_id).first()
    if not slip:
        raise HTTPException(status_code=404, detail="Deposit slip not found")
    if slip.status == "processed":
        raise HTTPException(status_code=400, detail="Slip already processed and validated")
    slip.override_reason = reason
    slip.override_approved_by = approved_by
    slip.override_at = _dt.utcnow()
    slip.status = "overridden"
    db.commit()
    db.refresh(slip)
    return {"message": "Override recorded", "slip_id": slip.id}

if __name__ == '__main__':
    uvicorn.run('app.main:app', host='0.0.0.0', port=int(os.getenv('PORT', 8000)), reload=True)


