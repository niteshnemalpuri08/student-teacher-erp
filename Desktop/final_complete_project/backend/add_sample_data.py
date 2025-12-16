import json
import os
import sys

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from flask import Flask
from backend.models import db, Teacher

# Create Flask app and initialize database
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(PROJECT_ROOT, "instance", "school_erp.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

def load_teachers():
    """Load teachers from JSON file into database"""
    with app.app_context():
        # Check if teachers already exist
        existing_count = Teacher.query.count()
        if existing_count > 0:
            print(f"Teachers already loaded: {existing_count} teachers found")
            return

        # Load teachers from JSON
        teachers_file = os.path.join(PROJECT_ROOT, 'backend', 'data', 'teachers.json')
        if not os.path.exists(teachers_file):
            print(f"Teachers file not found: {teachers_file}")
            return

        with open(teachers_file, 'r') as f:
            teachers_data = json.load(f)

        # Add teachers to database
        for teacher_data in teachers_data:
            teacher = Teacher(
                username=teacher_data['username'],
                password=teacher_data['password'],  # Plain text as per model
                name=teacher_data['name'],
                email=teacher_data['email'],
                department=teacher_data['department']
            )
            db.session.add(teacher)

        db.session.commit()
        print(f"Successfully loaded {len(teachers_data)} teachers into database")

if __name__ == "__main__":
    load_teachers()
