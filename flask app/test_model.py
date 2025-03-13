import pickle

# Try loading one model
with open('models/job_role_classifier1JR.pkl', 'rb') as file:
    model = pickle.load(file)
    print(model)
