# COMPARE PREDICITIVE POWER OF MODELS

import h2o
from h2o.estimators import H2ORandomForestEstimator
from h2o.grid.grid_search import H2OGridSearch

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
hyper_params = {'ntrees': [10, 20],
                'max_depth': [10, 20]}
search_criteria = {'strategy': 'RandomDiscrete',
                   'max_models': 36,
                   'seed': 1}

model = H2OGridSearch(model=H2ORandomForestEstimator,
                      hyper_params=hyper_params,
                      search_criteria=search_criteria)
model.train(x=x, y=y, training_frame=train, validation_frame=valid,
            nfolds=5)

# compare models
model.get_grid(sort_by='f1', decreasing=True)
