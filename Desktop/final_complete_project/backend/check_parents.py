from models import db, Parent
from server import app

with app.app_context():
    parents = Parent.query.all()
    print(f"Found {len(parents)} parents")
    for p in parents[:5]:
        print(f"{p.username}: {p.name} - student_roll: {p.student_roll}")
        print(f"Password hash exists: {bool(p.password_hash)}")
        print(f"Check password '24cse001': {p.check_password('24cse001')}")
        print("---")
