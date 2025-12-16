-- Extend SQLite database schema for internal marks, assignments, and student behavior
-- Linked to existing students table via roll foreign key

-- Table for internal marks/assessments
CREATE TABLE internal_marks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_roll TEXT NOT NULL,
    subject TEXT NOT NULL,
    assessment_type TEXT NOT NULL, -- e.g., 'midterm', 'quiz', 'project'
    marks REAL NOT NULL,
    max_marks REAL NOT NULL,
    date_assessed DATE NOT NULL,
    teacher_username TEXT NOT NULL,
    remarks TEXT,
    FOREIGN KEY (student_roll) REFERENCES students(roll),
    FOREIGN KEY (teacher_username) REFERENCES teachers(username)
);

-- Table for assignments
CREATE TABLE assignments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    subject TEXT NOT NULL,
    assigned_date DATE NOT NULL,
    due_date DATE NOT NULL,
    max_marks REAL NOT NULL,
    teacher_username TEXT NOT NULL,
    class_name TEXT NOT NULL,
    FOREIGN KEY (teacher_username) REFERENCES teachers(username)
);

-- Table for assignment submissions
CREATE TABLE assignment_submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    assignment_id INTEGER NOT NULL,
    student_roll TEXT NOT NULL,
    submission_date DATE,
    marks_obtained REAL,
    status TEXT NOT NULL DEFAULT 'pending', -- 'pending', 'submitted', 'graded'
    submission_file TEXT, -- file path or URL
    feedback TEXT,
    FOREIGN KEY (assignment_id) REFERENCES assignments(id),
    FOREIGN KEY (student_roll) REFERENCES students(roll)
);

-- Table for student behavior records
CREATE TABLE student_behavior (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_roll TEXT NOT NULL,
    behavior_type TEXT NOT NULL, -- 'positive', 'negative'
    description TEXT NOT NULL,
    points INTEGER NOT NULL, -- positive for good behavior, negative for bad
    date_recorded DATE NOT NULL,
    recorded_by TEXT NOT NULL, -- teacher username
    remarks TEXT,
    FOREIGN KEY (student_roll) REFERENCES students(roll),
    FOREIGN KEY (recorded_by) REFERENCES teachers(username)
);

-- Teacher mapping tables for comprehensive access control
CREATE TABLE teacher_departments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    teacher_username TEXT NOT NULL,
    department TEXT NOT NULL,
    assigned_date DATE NOT NULL DEFAULT CURRENT_DATE,
    is_active BOOLEAN NOT NULL DEFAULT 1,
    FOREIGN KEY (teacher_username) REFERENCES teachers(username),
    UNIQUE(teacher_username, department)
);

CREATE TABLE teacher_sections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    teacher_username TEXT NOT NULL,
    section TEXT NOT NULL,
    class_name TEXT NOT NULL,
    assigned_date DATE NOT NULL DEFAULT CURRENT_DATE,
    is_active BOOLEAN NOT NULL DEFAULT 1,
    FOREIGN KEY (teacher_username) REFERENCES teachers(username),
    UNIQUE(teacher_username, section, class_name)
);

CREATE TABLE teacher_subjects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    teacher_username TEXT NOT NULL,
    subject TEXT NOT NULL,
    class_name TEXT NOT NULL,
    section TEXT,
    assigned_date DATE NOT NULL DEFAULT CURRENT_DATE,
    is_active BOOLEAN NOT NULL DEFAULT 1,
    FOREIGN KEY (teacher_username) REFERENCES teachers(username),
    UNIQUE(teacher_username, subject, class_name, section)
);

-- Table for subject-wise attendance tracking
CREATE TABLE subject_attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_roll TEXT NOT NULL,
    subject TEXT NOT NULL,
    total_classes INTEGER NOT NULL DEFAULT 0,
    attended_classes INTEGER NOT NULL DEFAULT 0,
    attendance_percentage REAL NOT NULL DEFAULT 0.0,
    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_roll) REFERENCES students(roll),
    UNIQUE(student_roll, subject)
);

-- Indexes for better query performance
CREATE INDEX idx_internal_marks_student ON internal_marks(student_roll);
CREATE INDEX idx_internal_marks_subject ON internal_marks(subject);
CREATE INDEX idx_assignments_subject ON assignments(subject);
CREATE INDEX idx_assignments_class ON assignments(class_name);
CREATE INDEX idx_assignment_submissions_assignment ON assignment_submissions(assignment_id);
CREATE INDEX idx_assignment_submissions_student ON assignment_submissions(student_roll);
CREATE INDEX idx_student_behavior_student ON student_behavior(student_roll);
CREATE INDEX idx_student_behavior_type ON student_behavior(behavior_type);
CREATE INDEX idx_teacher_departments_teacher ON teacher_departments(teacher_username);
CREATE INDEX idx_teacher_departments_dept ON teacher_departments(department);
CREATE INDEX idx_teacher_sections_teacher ON teacher_sections(teacher_username);
CREATE INDEX idx_teacher_sections_section ON teacher_sections(section);
CREATE INDEX idx_teacher_subjects_teacher ON teacher_subjects(teacher_username);
CREATE INDEX idx_teacher_subjects_subject ON teacher_subjects(subject);
CREATE INDEX idx_subject_attendance_student ON subject_attendance(student_roll);
CREATE INDEX idx_subject_attendance_subject ON subject_attendance(subject);
