
<img src="cogs.png" alt="cogs" width=500px />

CoGS is intended to be a baseline search algorithm for the discovery of counterfactuals.
As the name suggests, CoGS is a *genetic algorithm*: it employs a population of candidate counterfactuals and does not require the machine learning model for which counterfactuals are sought to expose gradients.
CoGS is implemented in Python for ease of use, and heavily relies on NumPy for speed.

Colab example: https://colab.research.google.com/drive/1HQ4wcViJ5YV6w648yUtmiCoa2fGj4ftE

## Reference
Please consider citing the paper for which CoGS has been developed:
```
@misc{virgolin2022robustness,
      title={On the Robustness of Counterfactual Explanations to Adverse Perturbations}, 
      author={Marco Virgolin and Saverio Fracaros},
      year={2022},
      eprint={2201.09051},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


## Installation
Clone this repository, then `pip install .` from within it.

## Usage
CoGS is relatively simple to setup and run. 
Here's an example:
```python
cogs = Evolution(
        """ hyper-parameters of the problem (required!) """
        x=x,  # the starting point, for which the black-box model gives an undesired class prediction
        fitness_function=gower_fitness_function,  # a classic fitness function for counterfactual explanations
        fitness_function_kwargs={'blackbox':bbm,'desired_class': desired_class},  # bbm is the black-box model, these params are necessary
        feature_intervals=feature_intervals,  # intervals within which the search operates
        indices_categorical_features=indices_categorical_features,  # the indices of the features that are categorical
        plausibility_constraints=pcs, # can be "None" if no constraints need to be set
        """ hyper-parameters of the optimization (all optional) """
        evolution_type='classic', # the type of evolution, classic works well in general and is relatively fast to execute
        population_size=1000,   # how many candidate counterfactual examples to evolve simultaneously
        n_generations=100,  # number of iterations for the evolution
        selection_name='tournament_4', # selection pressure
        init_temperature=0.8, # how "far" from x we initialize
        num_features_mutation_strength=0.25, # strength of random mutations for numerical features
        num_features_mutation_strength_decay=0.5, # decay for the hyper-param. above
        num_features_mutation_strength_decay_generations=[50,75,90], # when to apply the decay
        """ other optional hyper-parameters """
        verbose=True  # logs progress at every generation 
)
cogs.run()
result = cogs.elite   # closest-found point to 'x' for which 'bbm.predict' returns 'desired_class'
```
The black-box model (`bbm`) can be anything, as long as it exposes a `predict` function, just like scikit-learn models do.
There's a full example in our notebook example.ipynb.

## Customization

### Fitness function
The quality of a candidate counterfactual `z` for the starting point `x` is called `fitness` (to be maximized).
The fitness implemented in CoGS is:
```python
-1*{0.5*gower_distance(z,x) + 0.5*L0(z,x) + int(bbm.predict(z)!=desired_class)}
```
which takes values between `(-inf,0)` (the closer to `0`, the better).
You can change the fitness function in `cogs/fitness.py` to any you like, as long as maximization is pursued. 
It is strongly recommended to evaluate the entire population of candidate counterfactuals using NumPy operations to maintain high efficiency.

### Crossover and mutation
Currently, CoGS implements the following operators to generate offspring counterfactuals:
* Crossover: Generates two offspring counterfactuals from two parent counterfactuals by swapping the feature values (called `genes`) of latter, uniformly at random (good under L0 requirements).
* Linear crossover: Works like the previous crossover for categorical features; for numerical features, the feature values of the offspring are random linear combinations of those of the parents (implemented but not used by default).
* Mutation: Generates an offspring counterfactual out of each parent. For categorical features, a new random category is sampled; for numerical features, the feature value of the parent is altered by a magnitude that depends on the interval of variation for that feature times the `num_features_mutation_strength` hyper-parameter.
Since mutation can result in feature values that are out-of-bounds for the data set at hand, corrections are implemented.

You can create your own crossover or mutation operator in `cogs/variation.py`. If your operator can generate values outside the intervals within which features take values, you should consider implement a correction mechanism (as currently present in `variation.generate_plausible_mutations`), or use the option `apply_fixes` in `fitness.gower_fitness_function`.

