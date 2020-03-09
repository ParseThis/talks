# Main Walkthrough
### Based on [MLFlow tutorial](https://www.mlflow.org/docs/latest/tutorials-and-examples/tutorial.html)


Walkthrough `examples/sklearn_elasticnet_wine/train.py` high-level
focus on mlflow tracking API:

```python
mlflow.start_run() # context manager for an run 
mlflow.log_param() # model hyperparamters
mlflow.log_metric() # model metrics 
```


run original model 
`python train.py`

should print 
```
RMSE: 0.7436470916334205
MAE: 0.6042761768399746
R2: 0.10601910075094545
```

run train.py multiple times
you'll end up writing this into train.py but bare with me

so we'll build 10 models where we change the l1 ratio
each time we increment the ratio by 0.1 in [0, 1] so effectively
in `range(0, 1, 0.1)`

`seq 0 0.1 1 | xargs -P4 -n1 -I{} python train.py 0.5 {}`


show ui with `mlflow ui` in the project dir/
compare `rmse` with `l1_ratio`

------------------- overtime ---------------------

# Packaging Sharable Models
We can package our model up to share with teams or downstream for model
serving.


show `MLProject` file

```yaml

name: tutorial-mlflow-talk

conda_env: conda.yaml

  main:
    parameters:
      alpha: {type: float, default: 0.5}
      l1_rati: {type: float, default: 0.5}
    command: "python train {alpha} {l1_ratio}
```

show `conda.yaml` 

```yaml
name: tutorial-mlflow-talk
channels:
  - defaults
dependencies:
  - python=3.6
  - scikit-learn=0.19.1
  - pip:
    - mlflow>-1.0
```

Run a model in the specified environment
from the project directory 

run `mlflow run . -P alpha=0.42`

go find the model artifact

show `model artifact` in ui

# Serving packaged model

Now we serve  that model with 
```bash
mlflow models serve \
	-m ~/projects/mlflow-talk/examples/sklearn_elasticnet_wine/mlruns/0/4dba2c548d1b47cd8e47d5bd7b54983d/artifacts/model \
	-p 1234

```


Our model is now served on `localhost:1234/invocations` we could query it with.

```bash
curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["alcohol", "chlorides", "citric acid", "density", "fixed acidity", "free sulfur dioxide", "pH", "residual sugar", "sulphates", "total sulfur dioxide", "volatile acidity"],"data":[[12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66]]}' http://127.0.0.1:1234/invocations 
```

So I could serve up my best model for testing
# What's More 

[See Tracking
API](https://www.mlflow.org/docs/latest/tutorials-and-examples/tutorial.html)

- Runs can be or grouped under Experiments 
- The backing store for the experiments dont have to be local (s3, MongoDB
  etc)
- Better API for Hyperaamater search
- Training in Kubernetes cluster
- Multistep workflows/Pipelines


