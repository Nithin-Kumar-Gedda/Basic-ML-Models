from pycaret.classification import * # type: ignore
from pycaret.datasets import get_data # type: ignore
import warnings

warnings.filterwarnings('ignore')

data = get_data('diabetes')
#print(data.head())
clf = setup(data=data, target= 'Class variable') # type: ignore

# Comparing all classification models
compare_models() # type: ignore

# Selecting the best model out of all models
best_model = create_model('lda') # type: ignore

# tuning model
tuned_model = tune_model(best_model) # type: ignore

# Plot a model
# plot_model(tuned_model, plot = 'confusion_matrix') # type: ignore
# plot_model(tuned_model, plot = 'class_report') # type: ignore
plot_model(tuned_model, plot = 'feature') # type: ignore