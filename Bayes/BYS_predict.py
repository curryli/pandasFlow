import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel

values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)), columns=['A', 'B', 'C', 'D', 'label'])
print values

train_data = values[:800]
predict_data = values[800:]

model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'label')])
model.fit(train_data)

predict_data = predict_data.copy()
predict_data.drop('label', axis=1, inplace=True)
y_pred = model.predict(predict_data)
print y_pred

y_prob = model.predict_probability(predict_data)
print y_prob