from fastapi import APIRouter, HTTPException,status
from models.case_model import CaseDocument
from schemas.case_schema import CaseCreate, CaseOut
from typing import List
from models.patient_model import PatientDocument 
from bson import ObjectId

router = APIRouter(prefix="/cases", tags=["Cases"])

@router.post("/",)
async def create_case(payload: CaseCreate, patient_id: str):
    patient = await PatientDocument.get(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    case = CaseDocument(**payload.dict(), patient_id=patient_id)
    await case.insert()

    patient.case_ids.append(str(case.id))
    await patient.save()

    return case

@router.get("/")
async def get_cases(patient_id: str):
    patient_id = ObjectId(patient_id)
    try:
        cases = await CaseDocument.find({"patient_id":patient_id}).to_list()
        if not cases:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No cases found for this patient"
            )
        return cases
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while fetching cases: {str(e)}"
        )
