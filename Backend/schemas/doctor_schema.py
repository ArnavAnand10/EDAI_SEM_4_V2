from pydantic import BaseModel, EmailStr
from typing import Optional

class DoctorCreate(BaseModel):
  
    email: EmailStr
    password: str 

class DoctorOut(BaseModel):
    id: str
   
    email: EmailStr
   

    class Config:
        from_attributes = True


class DoctorLogin(BaseModel):
    email: EmailStr
    password: str
    