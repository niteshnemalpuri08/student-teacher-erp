import json
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import db, Student, Parent, Teacher
from server import app

def fix_student_passwords():
    """Fix student passwords by hashing them properly"""
    with app.app_context():
        # Load student data
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(script_dir, 'data', 'students.json')
        with open(data_path, 'r') as f:
            students_data = json.load(f)

        print(f"Fixing passwords for {len(students_data)} students...")

        for student_data in students_data:
            username = student_data['username']
            plain_password = student_data['password']

            # Find student in database
            student = Student.query.filter_by(username=username).first()
            if student:
                # Set the password properly (this will hash it)
                student.set_password(plain_password)
                print(f"Fixed password for student: {username}")
            else:
                print(f"Student not found: {username}")

        # Commit changes
        db.session.commit()
        print("Student passwords fixed!")

def fix_parent_passwords():
    """Fix parent passwords - set to same as parent username"""
    with app.app_context():
        parents = Parent.query.all()
        print(f"Fixing passwords for {len(parents)} parents...")

        for parent in parents:
            # Set password to same as parent username
            parent.set_password(parent.username)
            print(f"Fixed password for parent: {parent.username}")

        db.session.commit()
        print("Parent passwords fixed!")

def fix_teacher_passwords():
    """Teachers use plain text passwords, no need to fix"""
    print("Teachers already use plain text passwords, no fix needed.")

if __name__ == "__main__":
    fix_student_passwords()
    fix_parent_passwords()
    fix_teacher_passwords()
    print("All passwords fixed!")
