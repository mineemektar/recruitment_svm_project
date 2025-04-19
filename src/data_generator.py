# data_generator.py
#
# This script generates synthetic applicant data for a hiring decision model.
# Each applicant has experience (in years), a technical test score, and a label
# indicating whether they would be hired based on a simple rule-based logic.
#
# random was used to generate numeric values for the applicants' experience years
# and technical test scores, which are the core features for the hiring prediction model.
# This decision was made because generating numerical data for these key attributes
# is essential for building a model that can predict hiring outcomes based on specific criteria.
#
# Faker is an optional library that could be used for generating realistic non-numeric data,
# such as names and emails. While it's not essential for the current goal of predicting
# hiring decisions, it could be useful if the dataset is expanded in the future to include
# more personal attributes or if there’s a need to make the data look more realistic
# in future versions of the project.


import pandas as pd
import numpy as np
from faker import Faker
import random
import os

# Initialize Faker for generating fake names
fake = Faker()

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def generate_applicant_data(num_samples=10000):
    """
    Generates synthetic applicant data for a software developer position.

    The labeling rule is:
    - If experience < 2 and score < 60 → Not Hired (label = 1)
    - Otherwise → Hired (label = 0)

    Parameters:
        num_samples (int): Number of applicants to generate. Default is 5000.

    Returns:
        pd.DataFrame: A DataFrame containing fake applicant data with columns:
            - name (str): Fake applicant name
            - experience_years (float): Years of software development experience (0–10)
            - technical_score (int): Score from a technical test (0–100)
            - label (int): Hiring decision (0 = hired, 1 = rejected)
    """
    applicants = []

    for _ in range(num_samples):
        # Random experience between 0 and 10 years
        experience_years = round(random.uniform(0, 10), 1)

        # Random technical test score between 0 and 100
        technical_score = random.randint(0, 100)

        # Rule-based labeling:
        # If experience < 2 years AND technical score < 60 → rejected (label = 1)
        if experience_years < 2 and technical_score < 60:
            label = 1  # Not hired
        else:
            label = 0  # Hired

        # Append the sample to the list
        applicants.append({
            'name': fake.name(),
            'experience_years': experience_years,
            'technical_score': technical_score,
            'label': label
        })

    df = pd.DataFrame(applicants)
    return df

if __name__ == "__main__":
    # Generate the data
    df = generate_applicant_data()

    # Print the first 5 rows to verify data generation
    print(df.head())

    # Create data directory if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(__file__), "../data"), exist_ok=True)

    # Save the dataset as a CSV file
    try:
        df.to_csv("../data/dataset.csv", index=False)
        print("Data successfully generated and saved to 'data/dataset.csv'")
    except Exception as e:
        print(f"Error while saving data: {e}")
