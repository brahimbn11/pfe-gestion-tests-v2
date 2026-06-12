from database import SessionLocal
from models import AssistantConversation, AssistantMessage, Incident, Solution

db = SessionLocal()

# delete child tables first (important)
db.query(AssistantMessage).delete()
db.query(AssistantConversation).delete()
db.query(Solution).delete()
db.query(Incident).delete()

db.commit()
db.close()

print("Database reset completed.")