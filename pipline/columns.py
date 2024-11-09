
mode_impute_columns = ['teacher_quality',
                        'parental_education_level',
                          'distance_from_home'
                     ]

outlier_columns = [
                   'tutoring_sessions']

cat_columns = ['age',
             'loan_limit',
             'Credit_Worthiness',
             'business_or_commercial',
             'Neg_ammortization',
             'interest_only',
             'lump_sum_payment',
             'occupancy_type',
             'submission_of_application',
             'co-applicant_credit_type',
             'loan_purpose',
             'loan_type',
             ]
mapping_dict = {
    'Positive': 2, 'Negative': 0,'Neutral':1,
    'Yes': 1, 'No': 0,
    'Low': 0, 'Medium': 1, 'High': 2,
    'Private': 1, 'Public': 0,
    'Near': 2, 'Moderate': 1, 'Far': 0,
    'High School': 0, 'College': 1, 'Postgraduate': 2,
    'Male': 0, 'Female': 1,
    0: 0, 1: 1, 2: 2  # Додаємо заміну для 0, 1, 2
}

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

normis = [
    'hours_studied', 
    'attendance', 
    'access_to_resources', 
    'sleep_hours', 
    'previous_scores', 
    'tutoring_sessions', 
    'teacher_quality', 
    'physical_activity', 
    'parental_education_level', 
    'distance_from_home',
    'parental_involvement',
    'motivation_level',
    'family_income'

]
y_column = ['exam_score'] # target variable

