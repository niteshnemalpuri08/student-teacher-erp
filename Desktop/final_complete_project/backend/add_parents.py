import json
import os
import sys

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from flask import Flask
from backend.models import db, Parent, Student

# Create Flask app and initialize database
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(PROJECT_ROOT, "instance", "school_erp.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

def load_parents():
    """Load parents from students data"""
    with app.app_context():
        # Check if parents already exist
        existing_count = Parent.query.count()
        if existing_count > 0:
            print(f"Parents already loaded: {existing_count} parents found")
            return

        # Load students from JSON
        students_file = os.path.join(PROJECT_ROOT, 'backend', 'data', 'students.json')
        if not os.path.exists(students_file):
            print(f"Students file not found: {students_file}")
            return

        with open(students_file, 'r') as f:
            students_data = json.load(f)

        # Add parents to database
        for student_data in students_data:
            parent_username = f"p{student_data['roll']}"
            parent = Parent(
                username=parent_username,
                name=f"Parent of {student_data['name']}",
                email=f"parent.{student_data['roll'].lower()}@example.com",
                student_roll=student_data['roll']
            )
            parent.set_password(student_data['username'])  # Same password as student username
            db.session.add(parent)

        db.session.commit()
        print(f"Successfully loaded {len(students_data)} parents into database")

if __name__ == "__main__":
    load_parents()
