# COMPARE PREDICITIVE POWER OF MODELS

import h2o
from h2o.estimators import H2ORandomForestEstimator

# connect to h2o
h2o.init()

# load data
file_path = 'data/PS_20174392719_1491204439457_log.csv'
df = h2o.import_file(file_path, destination_frame='df')

# train and validation split
df_sample = df.split_frame(ratios=[.1],
                           destination_frames=['df_sample', 'null'])[0]
train, valid = df_sample.split_frame(ratios=[.8],
                                     destination_frames=['train', 'valid'])
h2o.remove(['df', 'null'])

# Identify predictors and response
x = train.columns
y = "isFraud"
x.remove('isFraud')
x.remove('isFlaggedFraud')

# For binary classification, response should be a factor
train[y] = train[y].asfactor()
valid[y] = valid[y].asfactor()

# random forest
model = H2ORandomForestEstimator(ntrees=100, max_depth=20, nfolds=5)
model.train(x=x, y=y, training_frame=train, validation_frame=valid)
# see localhost:54321 for validation metrics
