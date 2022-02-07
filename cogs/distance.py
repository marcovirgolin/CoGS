import numpy as np

def gower_distance(Z, x, num_feature_ranges, indices_categorical_features=None):
  is_single_candidate = len(Z.shape) == 1
  if is_single_candidate:
    Z = Z.reshape((1,-1))

  l = Z.shape[1]

  D_c, D_n = np.zeros(shape = Z.shape[0]), np.zeros(shape = Z.shape[0])

  # categorical features  
  if indices_categorical_features and len(indices_categorical_features) > 0:
    D_c = np.sum( Z[:,indices_categorical_features] != x[indices_categorical_features], axis=1 )

    # also set numerical features
    is_numerical = np.ones(l, bool)
    is_numerical[indices_categorical_features] = False
    Z_num = Z[:,is_numerical]
    x_num = x[is_numerical]
  else:
    Z_num = Z
    x_num = x

  # numerical features contribution
  D_n = np.sum( np.divide( np.abs(Z_num - x_num), num_feature_ranges), axis=1)
  D = 1/l * (D_c + D_n)

  if is_single_candidate:
    D = D[0]

  return D
      
