import pandas as pd
from sklearn.model_selection import train_test_split
ds = pd.read_csv("StudentPerformanceFactors.csv")
ds.columns = ds.columns.str.lower()

X_train, X_test = train_test_split(ds, train_size=0.9)

X_columns =     ['hours_studied',
                'attendance', 
                'parental_involvement',
                'access_to_resources', 
                'extracurricular_activities',
                'sleep_hours',
                'previous_scores', 
                'motivation_level',
                'internet_access',
                'tutoring_sessions',
                'family_income',
                'teacher_quality',
                'school_type', 
                'peer_influence',
                'physical_activity',
                'learning_disabilities',
                'parental_education_level',
                'distance_from_home',
                'gender',
                ]

X_train.to_csv('train.csv', index=False)
X_test[X_columns].to_csv('new_input.csv', index=False)