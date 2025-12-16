import os
import sys

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from flask import Flask
from backend.models import db, Assignment, AssignmentSubmission, StudentBehavior

# Create Flask app and initialize database
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(PROJECT_ROOT, "instance", "school_erp.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

def check_sample_data():
    """Check if sample assignments and behavior records exist"""
    with app.app_context():
        assignments_count = Assignment.query.count()
        submissions_count = AssignmentSubmission.query.count()
        behavior_count = StudentBehavior.query.count()

        print(f"Assignments: {assignments_count}")
        print(f"Assignment Submissions: {submissions_count}")
        print(f"Behavior Records: {behavior_count}")

        if assignments_count > 0:
            print("✅ Sample assignments data found")
        else:
            print("❌ No assignments found")

        if submissions_count > 0:
            print("✅ Sample submissions data found")
        else:
            print("❌ No submissions found")

        if behavior_count > 0:
            print("✅ Sample behavior records found")
        else:
            print("❌ No behavior records found")

if __name__ == "__main__":
    check_sample_data()
