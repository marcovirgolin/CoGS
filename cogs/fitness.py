import numpy as np

from cogs.distance import *
from cogs.util import *


def gower_fitness_function(genes, x, blackbox, desired_class, feature_intervals, 
  indices_categorical_features=None, plausibility_constraints=None, 
  apply_fixes=False):

  is_single_candidate = len(genes.shape) == 1

  if is_single_candidate:
    genes = genes.reshape((1,-1))

  if apply_fixes: # not needed with current methods in variation
    genes = fix_features(genes, x, feature_intervals, 
      indices_categorical_features, plausibility_constraints)

  # compute distance, must be normalized to cap at 1
  num_feature_ranges = compute_ranges_numerical_features(feature_intervals, indices_categorical_features)
  gower_dist = gower_distance(genes, x, num_feature_ranges, indices_categorical_features)
  l_0 = 1/genes.shape[1] * np.sum(genes != x, axis=1)
  dist = .5*gower_dist + .5*l_0

  # then add additional penalty of +1 if the class is not the one that we want
  preds = blackbox.predict(genes)
  failed_preds = preds != desired_class
  dist += failed_preds.astype(float) # add 1 if the class is incorrect
  # we maximize the fitness
  fitness_values = -dist

  if is_single_candidate:
    fitness_values = fitness_values[0]

  return fitness_values
