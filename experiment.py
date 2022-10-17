# %% 
import time
import random
import warnings
import itertools
import operators
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from deap import tools, creator, base, gp, algorithms

warnings.filterwarnings('ignore')

# %%
train_start = "2022-03-01"
train_end = "2022-05-31"
test_start = "2022-06-01"
test_end = "2022-08-31"
adj_open = pd.read_parquet('data/raw_factor/adj_open.parquet')
adj_close = pd.read_parquet('data/raw_factor/adj_close.parquet')
adj_high = pd.read_parquet('data/raw_factor/adj_high.parquet')
adj_low = pd.read_parquet('data/raw_factor/adj_low.parquet')
volume = pd.read_parquet('data/raw_factor/volume.parquet')
label = pd.read_parquet('data/raw_factor/label.parquet')

# %%
adj_open_train = adj_open.loc[train_start:train_end].values
adj_close_train = adj_close.loc[train_start:train_end].values
adj_high_train = adj_high.loc[train_start:train_end].values
adj_low_train = adj_low.loc[train_start:train_end].values
volume_train = volume.loc[train_start:train_end].values
label_train = label.loc[train_start:train_end].values
adj_open_test = adj_open.loc[test_start:test_end].values
adj_close_test = adj_close.loc[test_start:test_end].values
adj_high_test = adj_high.loc[test_start:test_end].values
adj_low_test = adj_low.loc[test_start:test_end].values
volume_test = volume.loc[test_start:test_end].values
label_test = label.loc[test_start:test_end].values

# %%
pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(np.ndarray, 5), np.ndarray, "ARG")
pset.addPrimitive(operators.add, [np.ndarray, np.ndarray], np.ndarray)
pset.addPrimitive(operators.sub, [np.ndarray, np.ndarray], np.ndarray)
pset.addPrimitive(operators.mul, [np.ndarray, np.ndarray], np.ndarray)
pset.addPrimitive(operators.div, [np.ndarray, np.ndarray], np.ndarray)
pset.addEphemeralConstant("randint", lambda: random.randint(1, 10), int)
pset.addEphemeralConstant("rand100", lambda: random.uniform(1e-2, 1), float)
pset.renameArguments(ARG0='adj_open')
pset.renameArguments(ARG1='adj_high')
pset.renameArguments(ARG2='adj_low')
pset.renameArguments(ARG3='adj_close')
pset.renameArguments(ARG4='vol')

# %%
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# %%
def evaluate(individual):
    func: callable = toolbox.compile(individual)
    pred: np.ndarray = func(adj_open_train, adj_high_train, adj_low_train, adj_close_train, volume_train)
    if not isinstance(pred, np.ndarray):
        return np.nan,
    pred_demean = np.nan_to_num(pred - np.nanmean(pred, axis=1).repeat(pred.shape[1]).reshape(pred.shape[0], pred.shape[1]))
    label_demean = np.nan_to_num(label_train - np.nanmean(label_train, axis=1).repeat(
        label_train.shape[1]).reshape(label_train.shape[0], label_train.shape[1]))
    corrs = (pred_demean @ label_demean.T).diagonal() / (
        np.linalg.norm(pred_demean, axis=1) * np.linalg.norm(label_demean, axis=1))
    return np.abs(corrs.mean()),

toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# %%
def async_map(func, jobs):
    return Parallel(n_jobs=1, backend='threading')(delayed(func)(job) for job in jobs)
toolbox.register("map", async_map)

# %%
def main():
    random.seed(10)
    start_time = time.time()
    pop = toolbox.population(n=200)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("elapsed", lambda _: f"{time.time() - start_time:.2f}s")
    stats.register("avg", np.nanmean)
    stats.register("std", np.nanstd)
    stats.register("min", np.nanmin)
    stats.register("max", np.nanmax)

    algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 100, stats, halloffame=hof, verbose=True)

    return pop, stats, hof

_, _, hof = main()
for elite in hof:
    print(elite)
    pred_test = toolbox.compile(elite)(adj_open_test, adj_high_test, adj_low_test, adj_close_test, volume_test)
    corr = pd.DataFrame(pred_test).corrwith(pd.DataFrame(label_test), axis=1)
    print(f"Noraml IC on test Dataset: {corr.mean()}")
