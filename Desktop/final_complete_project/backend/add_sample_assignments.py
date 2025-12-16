import json
import os
import sys
from datetime import datetime, timedelta

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from flask import Flask
from backend.models import db, Assignment, AssignmentSubmission, StudentBehavior, Student, Teacher

# Create Flask app and initialize database
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(PROJECT_ROOT, "instance", "school_erp.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

def add_sample_assignments():
    """Add sample assignments and submissions"""
    with app.app_context():
        # Check if assignments already exist
        existing_count = Assignment.query.count()
        if existing_count > 0:
            print(f"Assignments already exist: {existing_count} assignments found")
            return

        # Get teachers and students
        teachers = Teacher.query.all()
        students = Student.query.all()

        if not teachers or not students:
            print("No teachers or students found. Please run add_sample_data.py and add_parents.py first.")
            return

        # Sample assignments data
        assignments_data = [
            {
                "title": "Mathematics Problem Set 1",
                "description": "Solve the quadratic equations and word problems in Chapter 5.",
                "subject": "Mathematics",
                "class_name": "CSE",
                "teacher_username": teachers[0].username,
                "due_date": datetime.now() + timedelta(days=7),
                "max_score": 100
            },
            {
                "title": "Physics Lab Report",
                "description": "Write a detailed lab report on the pendulum experiment including calculations and observations.",
                "subject": "Physics",
                "class_name": "CSE",
                "teacher_username": teachers[0].username,
                "due_date": datetime.now() + timedelta(days=5),
                "max_score": 50
            },
            {
                "title": "Chemistry Research Paper",
                "description": "Research and write about the properties of acids and bases. Include real-world applications.",
                "subject": "Chemistry",
                "class_name": "CSE",
                "teacher_username": teachers[0].username,
                "due_date": datetime.now() + timedelta(days=10),
                "max_score": 75
            },
            {
                "title": "Computer Science Algorithm Analysis",
                "description": "Analyze the time complexity of sorting algorithms and implement bubble sort.",
                "subject": "Computer Science",
                "class_name": "CSE",
                "teacher_username": teachers[0].username,
                "due_date": datetime.now() + timedelta(days=3),
                "max_score": 60
            },
            {
                "title": "English Literature Essay",
                "description": "Write a 500-word essay on the themes in Shakespeare's Romeo and Juliet.",
                "subject": "English",
                "class_name": "CSE",
                "teacher_username": teachers[0].username,
                "due_date": datetime.now() + timedelta(days=14),
                "max_score": 80
            }
        ]

        # Add assignments
        assignments = []
        for assignment_data in assignments_data:
            assignment = Assignment(**assignment_data)
            db.session.add(assignment)
            assignments.append(assignment)

        db.session.commit()

        # Add submissions for some students
        for assignment in assignments:
            # Submit for 80% of students
            submitted_students = students[:int(len(students) * 0.8)]

            for student in submitted_students:
                submission = AssignmentSubmission(
                    assignment_id=assignment.id,
                    student_roll=student.roll,
                    submitted_at=datetime.now() - timedelta(days=1),
                    status="graded",
                    score=round(60 + (40 * (student.avg_marks / 100)), 1)  # Score based on student performance
                )
                db.session.add(submission)

        db.session.commit()
        print(f"Successfully added {len(assignments)} assignments and {len(submitted_students) * len(assignments)} submissions")

def add_sample_behavior_records():
    """Add sample behavior records"""
    with app.app_context():
        # Check if behavior records already exist
        existing_count = StudentBehavior.query.count()
        if existing_count > 0:
            print(f"Behavior records already exist: {existing_count} records found")
            return

        # Get teachers and students
        teachers = Teacher.query.all()
        students = Student.query.all()

        if not teachers or not students:
            print("No teachers or students found. Please run add_sample_data.py and add_parents.py first.")
            return

        # Sample behavior records
        behavior_data = [
            {
                "student_roll": students[0].roll,
                "type": "Positive Participation",
                "description": "Excellent participation in class discussion and helped fellow students.",
                "points": 5,
                "recorded_by": teachers[0].username,
                "date_recorded": datetime.now() - timedelta(days=2)
            },
            {
                "student_roll": students[0].roll,
                "type": "Homework Completion",
                "description": "Consistently completing homework on time.",
                "points": 3,
                "recorded_by": teachers[0].username,
                "date_recorded": datetime.now() - timedelta(days=5)
            },
            {
                "student_roll": students[1].roll,
                "type": "Leadership",
                "description": "Took initiative in group project and led the team effectively.",
                "points": 4,
                "recorded_by": teachers[0].username,
                "date_recorded": datetime.now() - timedelta(days=3)
            },
            {
                "student_roll": students[1].roll,
                "type": "Discipline Issue",
                "description": "Talking during class without permission.",
                "points": -2,
                "recorded_by": teachers[0].username,
                "date_recorded": datetime.now() - timedelta(days=7)
            },
            {
                "student_roll": students[2].roll,
                "type": "Academic Excellence",
                "description": "Outstanding performance in mathematics quiz.",
                "points": 5,
                "recorded_by": teachers[0].username,
                "date_recorded": datetime.now() - timedelta(days=1)
            },
            {
                "student_roll": students[2].roll,
                "type": "Sports Achievement",
                "description": "Won first place in inter-class basketball tournament.",
                "points": 4,
                "recorded_by": teachers[0].username,
                "date_recorded": datetime.now() - timedelta(days=4)
            },
            {
                "student_roll": students[3].roll,
                "type": "Late Submission",
                "description": "Submitted assignment after due date.",
                "points": -1,
                "recorded_by": teachers[0].username,
                "date_recorded": datetime.now() - timedelta(days=6)
            },
            {
                "student_roll": students[4].roll,
                "type": "Volunteer Work",
                "description": "Helped organize school charity event.",
                "points": 3,
                "recorded_by": teachers[0].username,
                "date_recorded": datetime.now() - timedelta(days=8)
            }
        ]

        # Add behavior records
        for record_data in behavior_data:
            record = StudentBehavior(**record_data)
            db.session.add(record)

        db.session.commit()
        print(f"Successfully added {len(behavior_data)} behavior records")

if __name__ == "__main__":
    add_sample_assignments()
    add_sample_behavior_records()
