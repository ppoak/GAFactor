# %% 

import time
import random
import itertools
import operators
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from deap import tools, creator, base, gp, algorithms

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
adj_open_train = adj_open.loc[train_start:train_end]
adj_close_train = adj_close.loc[train_start:train_end]
adj_high_train = adj_high.loc[train_start:train_end]
adj_low_train = adj_low.loc[train_start:train_end]
volume_train = volume.loc[train_start:train_end]
label_train = label.loc[train_start:train_end]

# %%
pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(pd.DataFrame, 5), pd.DataFrame, "ARG")
pset.addPrimitive(operators.add, [pd.DataFrame, pd.DataFrame], pd.DataFrame)
pset.addPrimitive(operators.sub, [pd.DataFrame, pd.DataFrame], pd.DataFrame)
pset.addPrimitive(operators.mul, [pd.DataFrame, pd.DataFrame], pd.DataFrame)
pset.addPrimitive(operators.div, [pd.DataFrame, pd.DataFrame], pd.DataFrame)
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
def pearson_corr(x, y):
    res_x = x - np.nanmean(x)
    res_y = y - np.nanmean(y)
    return np.dot(res_x, res_y) / (np.linalg.norm(res_x) * np.linalg.norm(res_y))

def evaluate(individual):
    func = toolbox.compile(individual)
    pred = func(adj_open, adj_high, adj_low, adj_close, volume)
    if not isinstance(pred, pd.DataFrame):
        return np.nan,
    corrs = pred.corrwith(label, axis=1)
    return np.abs(corrs.mean()),

toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# %%
def async_map(func, jobs):
    return Parallel(n_jobs=-1, backend='threading')(delayed(func)(job) for job in jobs)
toolbox.register("map", async_map)

# %%
def main():
    random.seed(10)
    start_time = time.time()
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(10)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("elapsed", lambda _: f"{time.time() - start_time:.2f}s")
    stats.register("avg", np.nanmean)
    stats.register("std", np.nanstd)
    stats.register("min", np.nanmin)
    stats.register("max", np.nanmax)

    algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 40, stats, halloffame=hof, verbose=True)

    return pop, stats, hof

_, _, hof = main()
for elite in hof:
    print(elite)
