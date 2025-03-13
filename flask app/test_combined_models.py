import joblib

# Load the combined job category and job role models
job_category_model = joblib.load('models/job_category_classifier_combined.pkl')
job_role_model = joblib.load('models/job_role_classifier_combined.pkl')

# Access models and vectorizers
category_classifier = job_category_model['model']
category_vectorizer = job_category_model['vectorizer']

role_classifier = job_role_model['model']
role_vectorizer = job_role_model['vectorizer']

# Test if models are loaded correctly
print("Job Category Model and Vectorizer:")
print(category_classifier)
print(category_vectorizer)

print("\nJob Role Model and Vectorizer:")
print(role_classifier)
print(role_vectorizer)
