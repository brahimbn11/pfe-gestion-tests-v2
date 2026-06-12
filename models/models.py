from sqlalchemy import Column, Integer, String, Text, DateTime, Enum, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    nom = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    mot_de_passe = Column(String(255), nullable=False)
    role = Column(Enum("admin", "ingenieur", "technicien", name="role_enum"), default="technicien")
    discord_id = Column(String(50), unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    tests = relationship("Test", back_populates="user")
    incidents = relationship("Incident", back_populates="user")
    rapports = relationship("Rapport", back_populates="user")
    solutions = relationship("Solution", back_populates="user")


class Test(Base):
    __tablename__ = "tests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    nom_test = Column(String(200), nullable=False)
    equipement = Column(String(200), nullable=False)
    client = Column(String(200))
    resultat = Column(Enum("OK", "NOK", "EN_COURS", name="resultat_enum"), default="EN_COURS")
    date_test = Column(DateTime, default=datetime.utcnow)
    ingenieur = Column(String(100))
    rapport_ia = Column(Text)
    id_user = Column(Integer, ForeignKey("users.id"))

    user = relationship("User", back_populates="tests")
    incidents = relationship("Incident", back_populates="test")
    rapports = relationship("Rapport", back_populates="test")


class Incident(Base):
    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    titre = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    type_probleme = Column(String(100))
    equipement = Column(String(200))
    statut = Column(Enum("ouvert", "en_cours", "resolu", name="statut_enum"), default="ouvert")
    cause = Column(Text)
    solution = Column(Text)
    date_creation = Column(DateTime, default=datetime.utcnow)
    id_test = Column(Integer, ForeignKey("tests.id"))
    id_user = Column(Integer, ForeignKey("users.id"))
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=True)


    test = relationship("Test", back_populates="incidents")
    user = relationship("User", back_populates="incidents")
    rapports = relationship("Rapport", back_populates="incident")
    solutions = relationship("Solution", back_populates="incident")


class Rapport(Base):
    __tablename__ = "rapports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    titre = Column(String(200), nullable=False)
    contenu = Column(Text, nullable=False)
    type_rapport = Column(Enum("test", "incident", "analyse", name="type_rapport_enum"), default="test")
    date_generation = Column(DateTime, default=datetime.utcnow)
    id_test = Column(Integer, ForeignKey("tests.id"))
    id_incident = Column(Integer, ForeignKey("incidents.id"))
    id_user = Column(Integer, ForeignKey("users.id"))

    test = relationship("Test", back_populates="rapports")
    incident = relationship("Incident", back_populates="rapports")
    user = relationship("User", back_populates="rapports")


class Solution(Base):
    __tablename__ = "solutions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    titre = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    type_probleme = Column(String(100))
    equipement = Column(String(200))
    efficacite = Column(Integer, default=0)
    date_ajout = Column(DateTime, default=datetime.utcnow)
    id_incident = Column(Integer, ForeignKey("incidents.id"))
    id_user = Column(Integer, ForeignKey("users.id"))

    incident = relationship("Incident", back_populates="solutions")
    user = relationship("User", back_populates="solutions")

class AssistantConversation(Base):
    __tablename__ = "assistant_conversations"

    id = Column(Integer, primary_key=True, index=True)
    discord_user_id = Column(String(100), nullable=False)
    incident_id = Column(Integer, ForeignKey("incidents.id"), nullable=True)
    initial_question = Column(Text, nullable=False)
    rag_context = Column(Text)
    status = Column(
        Enum("active", "solved", "closed"),
        default="active"
    )

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )


class AssistantMessage(Base):
    __tablename__ = "assistant_messages"

    id = Column(Integer, primary_key=True, index=True)

    conversation_id = Column(
        Integer,
        ForeignKey("assistant_conversations.id"),
        nullable=False
    )

    role = Column(
        Enum("user", "assistant"),
        nullable=False
    )

    message = Column(Text, nullable=False)

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

class Category(Base):
    __tablename__ = "categories"

    id = Column(Integer, primary_key=True, index=True)
    nom = Column(String(100), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
