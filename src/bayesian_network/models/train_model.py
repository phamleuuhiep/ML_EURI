import os
import joblib
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch, MaximumLikelihoodEstimator, BIC

# Paths
PROCESSED_DATA_PATH = "data/processed/bayesian_network/"
TRAINED_MODEL_PATH = "models/experiments/bayesian_network/"

# Load processed data
def load_processed_data(filepath):
    return pd.read_pickle(filepath)

def build_bayes_network(df_discrete):
  # Structure learning (Hill Climb with BIC score)
  hc = HillClimbSearch(df_discrete)
  best_model = hc.estimate(scoring_method=BIC(df_discrete))

  # Create Bayesian Network with learned structure
  model = DiscreteBayesianNetwork(best_model.edges())

  # Parameter learning (estimate CPDs)
  model.fit(df_discrete, estimator=MaximumLikelihoodEstimator)

  return model

def main():
    df_discrete = load_processed_data(os.path.join(PROCESSED_DATA_PATH, "df_discrete.pkl"))

    model = build_bayes_network(df_discrete)

    print(model.edges())
    for cpd in model.get_cpds():
        print(cpd)

    joblib.dump(model, os.path.join(TRAINED_MODEL_PATH, "model.pkl"))

if __name__ == "__main__":
    main()