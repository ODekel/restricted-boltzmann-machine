import numpy as np
import numpy.typing as npt
import pandas as pd

from rbm import RestrictedBoltzmannMachine


def discretize(df: pd.DataFrame, column: str, bins: int):
    df.sort_values(by=column, inplace=True)
    df[column] = pd.qcut(df[column], bins, labels=False)


def int_to_bools_indicator(value: np.uint8, bits: int) -> npt.NDArray[np.bool]:
    bools = np.zeros(bits, dtype=np.bool)
    bools[value] = True
    return bools


int_to_bools_indicator_vec = np.vectorize(int_to_bools_indicator, otypes=[np.bool], excluded=[1], signature='()->(n)')


data = pd.read_csv('./IrisDataSet.csv', header=0)
data = data[['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Species']]
species = data['Species'].unique()
discretize(data, column='Sepal length', bins=len(species))
discretize(data, column='Sepal width', bins=len(species))
discretize(data, column='Petal length', bins=len(species))
discretize(data, column='Petal width', bins=len(species))
data['Species'] = (data['Species']
                   .replace({a_species: str(index) for index, a_species in enumerate(species)}).astype(int))
data = data.sample(frac=1).reset_index(drop=True)
data_np = data.to_numpy(dtype=np.uint8)

# Create numpy array of samples for RBM.
samples = int_to_bools_indicator_vec(data_np, len(species))
samples = samples.reshape(-1, 5 * len(species))
train_data = samples[:100]
test_data = samples[100:]

rbm = RestrictedBoltzmannMachine(hidden_size=8, visible_size=15, start_temp=1, temp_step=0, learn_rate=0.1)
for epoch in range(10):
    for sample in range(100):
        rbm.train(train_data[sample].reshape(-1, 1))

correct = 0
for sample in range(50):
    input_data = test_data[sample].reshape(-1, 1)
    input_mask = np.zeros_like(input_data, dtype=np.bool)
    input_mask[:4 * len(species)] = True
    output, output_mask = rbm.work(input_data[input_mask], input_mask, num_iterations=1000)
    if np.array_equal(input_data[output_mask], output[output_mask]):
        correct += 1
print(f"Accuracy: {correct}/50 = {correct / 50:.2%}")
