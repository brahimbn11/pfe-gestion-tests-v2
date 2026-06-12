from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import SessionLocal
from models.models import Incident
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class IncidentCreate(BaseModel):
    titre: str
    description: str
    type_probleme: Optional[str] = None
    equipement: Optional[str] = None
    statut: Optional[str] = "ouvert"
    cause: Optional[str] = None
    solution: Optional[str] = None
    id_test: Optional[int] = None
    id_user: Optional[int] = None


class IncidentUpdate(BaseModel):
    titre: Optional[str] = None
    description: Optional[str] = None
    type_probleme: Optional[str] = None
    equipement: Optional[str] = None
    statut: Optional[str] = None
    cause: Optional[str] = None
    solution: Optional[str] = None
    id_test: Optional[int] = None
    id_user: Optional[int] = None


@router.post("/incidents")
def create_incident(incident: IncidentCreate, db: Session = Depends(get_db)):
    new_incident = Incident(
        titre=incident.titre,
        description=incident.description,
        type_probleme=incident.type_probleme,
        equipement=incident.equipement,
        statut=incident.statut,
        cause=incident.cause,
        solution=incident.solution,
        id_test=incident.id_test,
        id_user=incident.id_user
    )

    db.add(new_incident)
    db.commit()
    db.refresh(new_incident)

    return new_incident


@router.get("/incidents")
def get_incidents(db: Session = Depends(get_db)):
    return db.query(Incident).all()


@router.get("/incidents/{incident_id}")
def get_incident(incident_id: int, db: Session = Depends(get_db)):
    incident = db.query(Incident).filter(Incident.id == incident_id).first()

    if not incident:
        raise HTTPException(status_code=404, detail="Incident non trouvé")

    return incident


@router.put("/incidents/{incident_id}")
def update_incident(
    incident_id: int,
    incident_update: IncidentUpdate,
    db: Session = Depends(get_db)
):
    incident = db.query(Incident).filter(Incident.id == incident_id).first()

    if not incident:
        raise HTTPException(status_code=404, detail="Incident non trouvé")

    if incident_update.titre is not None:
        incident.titre = incident_update.titre

    if incident_update.description is not None:
        incident.description = incident_update.description

    if incident_update.type_probleme is not None:
        incident.type_probleme = incident_update.type_probleme

    if incident_update.equipement is not None:
        incident.equipement = incident_update.equipement

    if incident_update.statut is not None:
        incident.statut = incident_update.statut

    if incident_update.cause is not None:
        incident.cause = incident_update.cause

    if incident_update.solution is not None:
        incident.solution = incident_update.solution

    if incident_update.id_test is not None:
        incident.id_test = incident_update.id_test

    if incident_update.id_user is not None:
        incident.id_user = incident_update.id_user

    db.commit()
    db.refresh(incident)

    return incident


@router.delete("/incidents/{incident_id}")
def delete_incident(incident_id: int, db: Session = Depends(get_db)):
    incident = db.query(Incident).filter(Incident.id == incident_id).first()

    if not incident:
        raise HTTPException(status_code=404, detail="Incident non trouvé")

    db.delete(incident)
    db.commit()

    return {"message": "Incident supprimé avec succès"}
