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
train = pd.read_parquet('data/train_dataset.parquet')
test = pd.read_parquet('data/test_dataset.parquet')
train_open, train_high, train_low, train_close, train_volume, train_label = (
    train['open'].unstack().values,
    train['high'].unstack().values,
    train['low'].unstack().values,
    train['close'].unstack().values,
    train['volume'].unstack().values,
    train['label'].unstack().values,
)
test_open, test_high, test_low, test_close, test_volume, test_label = (
    test['open'].unstack().values,
    test['high'].unstack().values,
    test['low'].unstack().values,
    test['close'].unstack().values,
    test['volume'].unstack().values,
    test['label'].unstack().values,
)

# %%
pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(np.ndarray, 5), np.ndarray, "ARG")
pset.addPrimitive(operators.add, [np.ndarray, np.ndarray], np.ndarray)
pset.addPrimitive(operators.add, [np.ndarray, float], np.ndarray)
pset.addPrimitive(operators.sub, [np.ndarray, np.ndarray], np.ndarray)
pset.addPrimitive(operators.sub, [np.ndarray, float], np.ndarray)
pset.addPrimitive(operators.mul, [np.ndarray, np.ndarray], np.ndarray)
pset.addPrimitive(operators.mul, [np.ndarray, float], np.ndarray)
pset.addPrimitive(operators.div, [np.ndarray, np.ndarray], np.ndarray)
pset.addPrimitive(operators.div, [np.ndarray, float], np.ndarray)
pset.addPrimitive(operators.sqrt, [np.ndarray], np.ndarray)
pset.addPrimitive(operators.ssqrt, [np.ndarray], np.ndarray)
pset.addPrimitive(operators.square, [np.ndarray], np.ndarray)
pset.addPrimitive(operators.raw, [float], float)
pset.addPrimitive(operators.raw, [int], int)
pset.addPrimitive(operators.mean, [np.ndarray, int], np.ndarray)
pset.addPrimitive(operators.std, [np.ndarray, int], np.ndarray)
pset.addPrimitive(operators.rank, [np.ndarray, int], np.ndarray)
pset.addPrimitive(operators.delay, [np.ndarray, int], np.ndarray)
pset.addPrimitive(operators.delta, [np.ndarray, int], np.ndarray)
pset.addPrimitive(operators.max_, [np.ndarray, int], np.ndarray)
pset.addPrimitive(operators.min_, [np.ndarray, int], np.ndarray)
pset.addPrimitive(operators.skew, [np.ndarray, int], np.ndarray)
pset.addPrimitive(operators.kurt, [np.ndarray, int], np.ndarray)
pset.addPrimitive(operators.var, [np.ndarray, int], np.ndarray)
pset.addPrimitive(operators.sum, [np.ndarray, int], np.ndarray)
pset.addPrimitive(operators.power, [np.ndarray, float], np.ndarray)
pset.addPrimitive(operators.power, [np.ndarray, int], np.ndarray)
pset.addEphemeralConstant("randfloat", lambda: random.uniform(1e-2, 10), float)
pset.addEphemeralConstant("randint", lambda: random.randint(1, 3), int)
pset.renameArguments(ARG0='open')
pset.renameArguments(ARG1='high')
pset.renameArguments(ARG2='low')
pset.renameArguments(ARG3='close')
pset.renameArguments(ARG4='vol')

# %%
creator.create("FitnessMax", base.Fitness, weights=(1.0, ))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=10)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# %%
def corr_dayavg(pred: np.ndarray, label: np.ndarray):
    pred_demean = np.nan_to_num(pred - np.nanmean(pred, axis=1).repeat(pred.shape[1]).reshape(pred.shape[0], pred.shape[1]))
    label_demean = np.nan_to_num(label - np.nanmean(label, axis=1).repeat(
        label.shape[1]).reshape(label.shape[0], label.shape[1]))
    corrs = (pred_demean @ label_demean.T).diagonal() / (
        np.linalg.norm(pred_demean, axis=1) * np.linalg.norm(label_demean, axis=1))
    return corrs.mean()

def mse_dayavg(pred: np.ndarray, label: np.ndarray):
    mse = np.nanmean((label - pred)**2, axis=1).mean()
    return mse

def count_nan(pred: np.ndarray):
    contain_nan = 0
    for v in pred:
        if np.count_nonzero(np.isnan(v)) > 0:
            contain_nan += 1
    if contain_nan > pred.shape[0] * 0.3:
        return np.inf
    return 0

def evaluate(individual):
    func: callable = toolbox.compile(individual)
    pred: np.ndarray = func(train_open, train_high, train_low, train_close, train_volume)
    if not isinstance(pred, np.ndarray):
        return np.nan,
    corr = corr_dayavg(pred, train_label)
    # mse = mse_dayavg(pred, label_train)
    # nan_punish = count_nan(pred)
    return np.abs(corr), 

toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=1, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# %%
def async_map(func, jobs):
    return Parallel(n_jobs=-1, backend='threading')(delayed(func)(job) for job in jobs)
toolbox.register("map", async_map)

# %%
def main():
    random.seed(10)
    start_time = time.time()
    pop = toolbox.population(n=200)
    hof = tools.HallOfFame(10)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("elapsed", lambda _: f"{time.time() - start_time:.2f}s")
    stats.register("avg", np.nanmean)
    stats.register("std", np.nanstd)
    stats.register("min", np.nanmin)
    stats.register("max", np.nanmax)

    algorithms.eaSimple(pop, toolbox, 0.5, 0.2, 50, stats, halloffame=hof, verbose=True)

    return pop, stats, hof

_, _, hof = main()
for elite in hof:
    train_pred = toolbox.compile(elite)(train_open, train_high, train_low, train_close, train_volume)
    test_pred = toolbox.compile(elite)(test_open, test_high, test_low, test_close, test_volume)
    train_corr = pd.DataFrame(train_pred).corrwith(pd.DataFrame(train_label), axis=1)
    test_corr = pd.DataFrame(test_pred).corrwith(pd.DataFrame(test_label), axis=1)
    print(f"Train IC: {train_corr.mean()}; Test IC: {test_corr.mean()}")
    filename = 'result/contradictory.txt' if (train_corr.mean() * test_corr.mean() < 0) \
        or np.isnan(train_corr.mean() * test_corr.mean()) else 'result/efficient.txt'
    with open(filename, 'a') as f:
        f.write(f'{elite}; train: {train_corr.mean():.4f}, test: {test_corr.mean():.4f}\n')
