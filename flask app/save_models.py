import pickle

# Load your trained models — ensure these are properly trained models
rf_classifier_categorization = ...  # Job role categorization model (RandomForest or other)
tfidf_vectorizer_categorization = ...  # TF-IDF vectorizer for job role categorization

rf_classifier_job_recommendation = ...  # Job category model
tfidf_vectorizer_job_recommendation = ...  # TF-IDF vectorizer for job recommendation

# Save job role categorization models
with open('models/job_role_classifier1JR.pkl', 'wb') as f:
    pickle.dump(rf_classifier_categorization, f)

with open('models/job_role_tfidf_vectorizer1.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer_categorization, f)

# Combine and save job category models into a single file
job_category_model = {
    'model': rf_classifier_job_recommendation,
    'vectorizer': tfidf_vectorizer_job_recommendation
}

with open('models/job_category_classifier_combined.pkl', 'wb') as f:
    pickle.dump(job_category_model, f)

print("Models have been saved successfully.")
