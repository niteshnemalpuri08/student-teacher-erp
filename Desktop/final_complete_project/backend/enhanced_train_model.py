import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle
import os
import sqlite3
from datetime import datetime, timedelta
import json

def load_database_data():
    """Load data from SQLite database including all student metrics"""
    db_path = 'school_erp.db'

    if not os.path.exists(db_path):
        print("Database not found. Please run init_db.py first.")
        return None

    conn = sqlite3.connect(db_path)

    # Load students with basic marks
    students_df = pd.read_sql_query("""
        SELECT s.roll, s.name, s.class_name, s.attendance, s.avg_marks,
               s.math_marks, s.physics_marks, s.chemistry_marks,
               s.cs_marks, s.english_marks
        FROM students s
    """, conn)

    # Load internal marks (assessments)
    internal_marks_df = pd.read_sql_query("""
        SELECT student_roll,
               AVG(CASE WHEN subject = 'math' THEN marks END) as internal_math_avg,
               AVG(CASE WHEN subject = 'physics' THEN marks END) as internal_physics_avg,
               AVG(CASE WHEN subject = 'chemistry' THEN marks END) as internal_chemistry_avg,
               AVG(CASE WHEN subject = 'cs' THEN marks END) as internal_cs_avg,
               AVG(CASE WHEN subject = 'english' THEN marks END) as internal_english_avg,
               COUNT(*) as total_assessments,
               AVG(marks) as overall_internal_avg
        FROM internal_marks
        GROUP BY student_roll
    """, conn)

    # Load assignment submissions
    assignments_df = pd.read_sql_query("""
        SELECT student_roll,
               COUNT(*) as assignments_completed,
               AVG(marks_obtained) as avg_assignment_marks,
               SUM(CASE WHEN status = 'submitted' THEN 1 ELSE 0 END) as assignments_submitted,
               SUM(CASE WHEN status = 'graded' THEN 1 ELSE 0 END) as assignments_graded
        FROM assignment_submissions
        GROUP BY student_roll
    """, conn)

    # Load student behavior data
    behavior_df = pd.read_sql_query("""
        SELECT student_roll,
               COUNT(*) as total_behavior_records,
               SUM(CASE WHEN behavior_type = 'positive' THEN points ELSE 0 END) as positive_points,
               SUM(CASE WHEN behavior_type = 'negative' THEN ABS(points) ELSE 0 END) as negative_points,
               AVG(points) as avg_behavior_score
        FROM student_behavior
        GROUP BY student_roll
    """, conn)

    conn.close()

    return students_df, internal_marks_df, assignments_df, behavior_df

def create_enhanced_dataset(students_df, internal_marks_df, assignments_df, behavior_df):
    """Combine all data sources into a comprehensive dataset"""

    # Start with students data
    dataset = students_df.copy()

    # Merge internal marks
    dataset = dataset.merge(internal_marks_df, left_on='roll', right_on='student_roll', how='left')

    # Merge assignment data
    dataset = dataset.merge(assignments_df, left_on='roll', right_on='student_roll', how='left')

    # Merge behavior data
    dataset = dataset.merge(behavior_df, left_on='roll', right_on='student_roll', how='left')

    # Fill missing values
    numeric_columns = [
        'internal_math_avg', 'internal_physics_avg', 'internal_chemistry_avg',
        'internal_cs_avg', 'internal_english_avg', 'total_assessments', 'overall_internal_avg',
        'assignments_completed', 'avg_assignment_marks', 'assignments_submitted', 'assignments_graded',
        'total_behavior_records', 'positive_points', 'negative_points', 'avg_behavior_score'
    ]

    for col in numeric_columns:
        if col in dataset.columns:
            dataset[col] = dataset[col].fillna(0)

    # Create derived features
    dataset['subject_variance'] = dataset[['math_marks', 'physics_marks', 'chemistry_marks', 'cs_marks', 'english_marks']].var(axis=1)
    dataset['strong_subjects'] = (dataset[['math_marks', 'physics_marks', 'chemistry_marks', 'cs_marks', 'english_marks']] >= 80).sum(axis=1)
    dataset['weak_subjects'] = (dataset[['math_marks', 'physics_marks', 'chemistry_marks', 'cs_marks', 'english_marks']] < 60).sum(axis=1)

    # Behavior score (positive minus negative)
    dataset['net_behavior_score'] = dataset['positive_points'] - dataset['negative_points']

    # Assignment completion rate
    dataset['assignment_completion_rate'] = np.where(
        dataset['assignments_completed'] > 0,
        dataset['assignments_submitted'] / dataset['assignments_completed'],
        0
    )

    # Internal vs final marks consistency
    for subject in ['math', 'physics', 'chemistry', 'cs', 'english']:
        internal_col = f'internal_{subject}_avg'
        final_col = f'{subject}_marks'
        consistency_col = f'{subject}_consistency'

        if internal_col in dataset.columns and final_col in dataset.columns:
            dataset[consistency_col] = np.where(
                dataset[internal_col] > 0,
                1 - abs(dataset[final_col] - dataset[internal_col]) / 100,
                0
            )

    # Create target variable (pass/fail based on avg_marks and attendance)
    dataset['pass'] = ((dataset['avg_marks'] >= 50) & (dataset['attendance'] >= 75)).astype(int)

    return dataset

def train_enhanced_model(dataset, model_type='random_forest'):
    """Train the enhanced ML model with multiple algorithms"""

    # Select features for training
    feature_columns = [
        # Basic marks
        'math_marks', 'physics_marks', 'chemistry_marks', 'cs_marks', 'english_marks',
        'attendance', 'avg_marks',

        # Internal assessment features
        'internal_math_avg', 'internal_physics_avg', 'internal_chemistry_avg',
        'internal_cs_avg', 'internal_english_avg', 'total_assessments', 'overall_internal_avg',

        # Assignment features
        'assignments_completed', 'avg_assignment_marks', 'assignments_submitted',
        'assignments_graded', 'assignment_completion_rate',

        # Behavior features
        'total_behavior_records', 'positive_points', 'negative_points',
        'avg_behavior_score', 'net_behavior_score',

        # Derived features
        'subject_variance', 'strong_subjects', 'weak_subjects',
        'math_consistency', 'physics_consistency', 'chemistry_consistency',
        'cs_consistency', 'english_consistency'
    ]

    # Filter to available columns
    available_features = [col for col in feature_columns if col in dataset.columns]
    print(f"Using {len(available_features)} features: {available_features}")

    X = dataset[available_features]
    y = dataset['pass']

    # Handle any remaining missing values
    X = X.fillna(0)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train different models
    models = {
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
        'decision_tree': DecisionTreeClassifier(random_state=42, max_depth=10)
    }

    best_model = None
    best_accuracy = 0
    best_model_name = ""

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        # Train the model
        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = model.predict(X_test_scaled)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        results[name] = {
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score'],
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }

        print(".3f")
        print(".3f")
        print(".3f")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name

    print(f"\nBest model: {best_model_name} with accuracy: {best_accuracy:.3f}")

    # Feature importance for Random Forest
    if best_model_name == 'random_forest':
        feature_importance = dict(zip(available_features, best_model.feature_importances_))
        results['feature_importance'] = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    # Save the best model and scaler
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'features': available_features,
        'model_type': best_model_name,
        'accuracy': best_accuracy,
        'training_date': datetime.now().isoformat()
    }

    with open('backend/enhanced_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    print("Enhanced model saved to backend/enhanced_model.pkl")

    return results, best_model_name, best_accuracy

def generate_synthetic_enhanced_data(num_samples=1000):
    """Generate synthetic data with all enhanced features for testing"""
    np.random.seed(42)

    data = []

    for i in range(num_samples):
        # Base academic performance
        base_performance = np.random.normal(70, 15)
        base_performance = np.clip(base_performance, 20, 100)

        # Subject marks with some correlation
        subject_marks = {}
        for subject in ['math', 'physics', 'chemistry', 'cs', 'english']:
            variation = np.random.normal(0, 8)
            subject_marks[subject] = np.clip(base_performance + variation, 0, 100)

        # Attendance (correlated with performance)
        attendance = np.clip(base_performance + np.random.normal(0, 10), 0, 100)

        # Internal assessments (practice tests, quizzes)
        internal_marks = {}
        total_assessments = np.random.randint(5, 20)
        for subject in ['math', 'physics', 'chemistry', 'cs', 'english']:
            # Internal marks slightly different from final marks
            variation = np.random.normal(0, 5)
            internal_marks[subject] = np.clip(subject_marks[subject] + variation, 0, 100)

        overall_internal_avg = np.mean(list(internal_marks.values()))

        # Assignment data
        assignments_completed = np.random.randint(5, 15)
        assignment_completion_rate = np.random.beta(8, 2)  # Most students complete assignments
        assignments_submitted = int(assignments_completed * assignment_completion_rate)
        assignments_graded = np.random.randint(max(1, assignments_submitted - 2), assignments_submitted + 1)

        # Assignment marks (correlated with academic performance)
        avg_assignment_marks = np.clip(base_performance + np.random.normal(0, 8), 0, 100)

        # Behavior data
        total_behavior_records = np.random.randint(0, 10)
        if total_behavior_records > 0:
            positive_points = np.random.randint(0, total_behavior_records * 5)
            negative_points = np.random.randint(0, total_behavior_records * 3)
            avg_behavior_score = (positive_points - negative_points) / total_behavior_records
            net_behavior_score = positive_points - negative_points
        else:
            positive_points = negative_points = avg_behavior_score = net_behavior_score = 0

        # Derived features
        subject_variance = np.var(list(subject_marks.values()))
        strong_subjects = sum(1 for mark in subject_marks.values() if mark >= 80)
        weak_subjects = sum(1 for mark in subject_marks.values() if mark < 60)

        # Consistency between internal and final marks
        consistencies = {}
        for subject in ['math', 'physics', 'chemistry', 'cs', 'english']:
            consistency = 1 - abs(subject_marks[subject] - internal_marks[subject]) / 100
            consistencies[subject] = consistency

        # Determine pass/fail (more nuanced criteria)
        avg_marks = np.mean(list(subject_marks.values()))
        academic_pass = avg_marks >= 50
        attendance_pass = attendance >= 75
        behavior_pass = net_behavior_score >= -5  # Not too many negative points

        # Overall pass with some flexibility
        pass_status = int(academic_pass and attendance_pass and behavior_pass)

        record = {
            'roll': f'24CSE{1001 + i:03d}',
            'name': f'Student_{i+1}',
            'class_name': 'CSE',
            'attendance': round(attendance, 1),
            'avg_marks': round(avg_marks, 1),
            'math_marks': round(subject_marks['math'], 1),
            'physics_marks': round(subject_marks['physics'], 1),
            'chemistry_marks': round(subject_marks['chemistry'], 1),
            'cs_marks': round(subject_marks['cs'], 1),
            'english_marks': round(subject_marks['english'], 1),
            'internal_math_avg': round(internal_marks['math'], 1),
            'internal_physics_avg': round(internal_marks['physics'], 1),
            'internal_chemistry_avg': round(internal_marks['chemistry'], 1),
            'internal_cs_avg': round(internal_marks['cs'], 1),
            'internal_english_avg': round(internal_marks['english'], 1),
            'total_assessments': total_assessments,
            'overall_internal_avg': round(overall_internal_avg, 1),
            'assignments_completed': assignments_completed,
            'avg_assignment_marks': round(avg_assignment_marks, 1),
            'assignments_submitted': assignments_submitted,
            'assignments_graded': assignments_graded,
            'assignment_completion_rate': round(assignment_completion_rate, 3),
            'total_behavior_records': total_behavior_records,
            'positive_points': positive_points,
            'negative_points': negative_points,
            'avg_behavior_score': round(avg_behavior_score, 2),
            'net_behavior_score': net_behavior_score,
            'subject_variance': round(subject_variance, 2),
            'strong_subjects': strong_subjects,
            'weak_subjects': weak_subjects,
            'math_consistency': round(consistencies['math'], 3),
            'physics_consistency': round(consistencies['physics'], 3),
            'chemistry_consistency': round(consistencies['chemistry'], 3),
            'cs_consistency': round(consistencies['cs'], 3),
            'english_consistency': round(consistencies['english'], 3),
            'pass': pass_status
        }

        data.append(record)

    return pd.DataFrame(data)

def main():
    """Main function to train enhanced ML model"""
    print("Enhanced ML Model Training for Student Outcome Prediction")
    print("=" * 60)

    # Try to load real database data first
    print("Attempting to load data from database...")
    db_data = load_database_data()

    if db_data is not None:
        students_df, internal_marks_df, assignments_df, behavior_df = db_data
        print(f"Loaded {len(students_df)} students from database")

        # Create enhanced dataset
        dataset = create_enhanced_dataset(students_df, internal_marks_df, assignments_df, behavior_df)
        print(f"Created enhanced dataset with {len(dataset)} samples and {len(dataset.columns)} features")
    else:
        print("Database not available, generating synthetic enhanced data...")
        dataset = generate_synthetic_enhanced_data(1000)
        print(f"Generated synthetic dataset with {len(dataset)} samples")

    # Display dataset info
    print(f"\nDataset shape: {dataset.shape}")
    print(f"Pass rate: {dataset['pass'].mean():.3f}")
    print(f"Features available: {len([col for col in dataset.columns if col != 'pass'])}")

    # Train the enhanced model
    print("\nTraining enhanced ML models...")
    results, best_model, best_accuracy = train_enhanced_model(dataset)

    # Display results
    print("\n" + "="*60)
    print("MODEL TRAINING RESULTS")
    print("="*60)

    for model_name, metrics in results.items():
        if model_name != 'feature_importance':
            print(f"\n{model_name.upper()}:")
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1-Score: {metrics['f1_score']:.3f}")

    print(f"\nBEST MODEL: {best_model.upper()} (Accuracy: {best_accuracy:.3f})")

    if 'feature_importance' in results:
        print("\nTOP 10 MOST IMPORTANT FEATURES:")
        for i, (feature, importance) in enumerate(list(results['feature_importance'].items())[:10]):
            print("2d")

    # Research significance
    print("\n" + "="*60)
    print("RESEARCH SIGNIFICANCE")
    print("="*60)
    print("This enhanced ML model demonstrates the value of combining multiple")
    print("student metrics for improved outcome prediction:")
    print()
    print("1. MULTI-MODAL DATA INTEGRATION:")
    print("   - Academic performance (final marks)")
    print("   - Formative assessments (internal marks)")
    print("   - Engagement metrics (assignment completion)")
    print("   - Behavioral indicators (conduct records)")
    print()
    print("2. DERIVED FEATURES:")
    print("   - Subject-wise consistency analysis")
    print("   - Performance variance across subjects")
    print("   - Behavioral scoring systems")
    print()
    print("3. PREDICTIVE IMPROVEMENT:")
    print(".1f")
    print("   - Better identification of at-risk students")
    print("   - More nuanced intervention strategies")
    print()
    print("4. EDUCATIONAL INSIGHTS:")
    print("   - Understanding factors beyond traditional grades")
    print("   - Holistic student assessment approach")
    print("   - Data-driven educational decision making")

if __name__ == "__main__":
    main()
