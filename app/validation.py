from typing import Dict
from sqlalchemy.orm import Session
from datetime import datetime, date
from .models import Collection, DepositSlip

class DepositSlipValidator:
    def __init__(self):
        self.validation_rules = {
            'amount_tolerance': 0.01,
            'date_range_days': 30,
            'min_amount': 0.01,
            'max_amount': 1000000.00,
        }

    def validate_slip_data(
        self,
        ocr_result: Dict,
        collection_id: int,
        manual_amount: float = None,
        manual_date: str = None,
        db: Session = None
    ) -> Dict:
        print(f"✅ DepositSlipValidator.validate_slip_data: Starting validation for collection {collection_id}")
        errors = []
        warnings = []
        collection = db.query(Collection).filter(Collection.id == collection_id).first()
        if not collection:
            print(f"❌ VALIDATION: Collection {collection_id} not found")
            errors.append("Collection record not found")
            return {'is_valid': False, 'errors': errors, 'warnings': warnings}
        print(f"✅ VALIDATION: Collection found - {collection.id}: {collection.amount}")

        amount_validation = self.validate_amount(
            ocr_result.get('amount'),
            manual_amount,
            collection.amount
        )
        if not amount_validation['is_valid']:
            errors.extend(amount_validation['errors'])
        warnings.extend(amount_validation.get('warnings', []))

        date_validation = self.validate_date(
            ocr_result.get('date'),
            manual_date,
            collection.date
        )
        if not date_validation['is_valid']:
            errors.extend(date_validation['errors'])
        warnings.extend(date_validation.get('warnings', []))

        if ocr_result.get('confidence', 0) < 0.7:
            warnings.append(f"Low OCR confidence: {ocr_result.get('confidence', 0):.2f}")

        existing_slip = db.query(DepositSlip).filter(
            DepositSlip.collection_id == collection_id
        ).first()
        if existing_slip:
            errors.append("Collection already has a deposit slip")

        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'collection_amount': collection.amount,
            'extracted_amount': ocr_result.get('amount'),
            'manual_amount': manual_amount
        }

    def validate_amount(self, ocr_amount: float, manual_amount: float, collection_amount: float) -> Dict:
        errors = []
        warnings = []
        deposit_amount = manual_amount if manual_amount is not None else ocr_amount
        if deposit_amount is None:
            errors.append("No amount found in deposit slip")
            return {'is_valid': False, 'errors': errors}

        if deposit_amount < self.validation_rules['min_amount']:
            errors.append(f"Amount too small: ${deposit_amount}")
        if deposit_amount > self.validation_rules['max_amount']:
            errors.append(f"Amount too large: ${deposit_amount}")

        amount_diff = abs(deposit_amount - collection_amount)
        if amount_diff > self.validation_rules['amount_tolerance']:
            if manual_amount is not None:
                errors.append(f"Manual amount ${deposit_amount} doesn't match collection ${collection_amount}")
            else:
                warnings.append(f"OCR amount ${deposit_amount} doesn't match collection ${collection_amount}")

        if ocr_amount and manual_amount and abs(ocr_amount - manual_amount) > self.validation_rules['amount_tolerance']:
            warnings.append(f"OCR amount ${ocr_amount} differs from manual amount ${manual_amount}")

        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }

    def validate_date(self, ocr_date: str, manual_date: str, collection_date: date) -> Dict:
        errors = []
        warnings = []
        try:
            if manual_date:
                deposit_date = datetime.strptime(manual_date, '%Y-%m-%d').date()
            elif ocr_date:
                deposit_date = datetime.strptime(ocr_date, '%Y-%m-%d').date()
            else:
                errors.append("No date found in deposit slip")
                return {'is_valid': False, 'errors': errors}
        except ValueError as e:
            errors.append(f"Invalid date format: {e}")
            return {'is_valid': False, 'errors': errors}

        if deposit_date < collection_date:
            errors.append(f"Deposit date {deposit_date} is before collection date {collection_date}")

        date_diff = (deposit_date - collection_date).days
        if date_diff > self.validation_rules['date_range_days']:
            warnings.append(f"Deposit date is {date_diff} days after collection")

        if ocr_date and manual_date:
            try:
                ocr_date_parsed = datetime.strptime(ocr_date, '%Y-%m-%d').date()
                manual_date_parsed = datetime.strptime(manual_date, '%Y-%m-%d').date()
                if ocr_date_parsed != manual_date_parsed:
                    warnings.append(f"OCR date {ocr_date} differs from manual date {manual_date}")
            except ValueError:
                pass

        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }


