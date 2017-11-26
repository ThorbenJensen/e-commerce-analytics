# COMPARE PREDICITIVE POWER OF MODELS

import h2o

# connect to h2o
h2o.init()

# load data
file_path = 'data/PS_20174392719_1491204439457_log.csv'
h2o.import_file(file_path)

# inspect data
