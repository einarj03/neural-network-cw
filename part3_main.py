import pandas as pd
from part3_pricing_model import PricingModel
from sklearn.model_selection import train_test_split

data = pd.read_csv("part3_training_data.csv")
X_raw = data.drop(columns=["claim_amount", "made_claim"])
y_raw = data["made_claim"]
claims_raw = data['claim_amount']
test_ratio = 0.2
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=test_ratio)


num_features = 13
pricing_model = PricingModel(epoch=100, batchsize=128, learnrate=0.01, neurons=9, num_features=num_features)

X_clean = pricing_model._preprocessor(X_raw)

y_raw[y_raw != 0].shape[0] / y_raw.shape[0]

pricing_model.fit(X_train, y_train, claims_raw)

y_pred = pricing_model.predict_claim_probability(X_test)
print(y_pred)
print(y_pred[y_pred > 0.1])

pricing_model.evaluate_architecture(X_test, y_test)