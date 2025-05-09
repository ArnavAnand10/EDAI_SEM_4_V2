from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from models.doctor_model import DoctorDocument  # Assuming you have the DoctorDocument model

uri = "mongodb+srv://arnavanand710:anand@cluster0.ikgfxgh.mongodb.net/?appName=Cluster0"

def connectToDB():
    client = MongoClient(uri, server_api=ServerApi('1'))

    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")


    except Exception as e:
        print(f"Error: {e}")

connectToDB()
