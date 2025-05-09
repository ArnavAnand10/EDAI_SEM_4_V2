from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from beanie import PydanticObjectId
from bson import ObjectId

from models.patient_model import PatientDocument
from schemas.patient_schema import PatientCreate, PatientOut
from typing import List
from models.doctor_model import DoctorDocument

router = APIRouter(prefix="/patients",tags=["Patients"])
from fastapi import APIRouter, HTTPException, status, Query
from fastapi.responses import JSONResponse


@router.post("/", status_code=status.HTTP_200_OK)
async def create_patient(payload: PatientCreate, doctor_id: str):
    print("Patient API called", payload)
    print("Doctor ID:", doctor_id)

    doctor = await DoctorDocument.get(doctor_id)
    if not doctor:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Doctor not found"
        )

    try:
        patient = PatientDocument(**payload.dict(), doctor_id=doctor.id)
        await patient.insert()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create patient: {str(e)}"
        )

    try:
        await doctor.update({"$push": {"patients": patient.id}})
    except Exception as e:
        await patient.delete()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Patient created but failed to link to doctor: {str(e)}"
        )

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=PatientOut  (**{
            **patient.dict(),
            "id": str(patient.id),
            "doctor_id": str(patient.doctor_id)
        }).dict()
    )


@router.get("/")
async def get_patients(doctor_id: str):
    doctor_id = ObjectId(doctor_id)
    try:
        patients = await PatientDocument.find({"doctor_id":doctor_id}).to_list()
        if not patients:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No patients found for this doctor"
            )
        return patients
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while fetching patients: {str(e)}"
        )