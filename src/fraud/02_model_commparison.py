# COMPARE PREDICITIVE POWER OF MODELS

import h2o
from h2o.automl import H2OAutoML

# connect to h2o
h2o.init()

# load data
file_path = 'data/PS_20174392719_1491204439457_log.csv'
df = h2o.import_file(file_path, destination_frame='df')
df = df.split_frame(ratios=[.1])[0]

# inspect data
df.describe()

# train and validation split
train, test, valid = df.split_frame(ratios=[.7, .15])

# Identify predictors and response
x = train.columns
y = "isFraud"
x.remove('isFraud')
x.remove('isFlaggedFraud')

# For binary classification, response should be a factor
train[y] = train[y].asfactor()
test[y] = test[y].asfactor()

# Run AutoML for 30 seconds
aml = H2OAutoML(max_runtime_secs = 60*20)
aml.train(x = x, y = y,
          training_frame = train,
          leaderboard_frame = test)
