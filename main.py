from fastapi import FastAPI
from database import engine
from models.models import Base
from routes.users import router as user_router
from routes import tests
from routes import incidents
from routes import reports
from routes import rag
from routes import assistant_chat

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.include_router(user_router)
app.include_router(tests.router)
app.include_router(incidents.router)
app.include_router(reports.router)
app.include_router(rag.router)
app.include_router(assistant_chat.router)


@app.get("/")
def root():
    return {"message": "API PFE fonctionne"}

