import numpy as np
import numpy.typing as npt


def _sigmoid(x: npt.NDArray) -> npt.NDArray:
    return 1 / (1 + np.exp(-x))


class RestrictedBoltzmannMachine:
    def __init__(self, hidden_size: int, visible_size: int, start_temp: float, temp_step: float, learn_rate: float):
        self._rng = np.random.default_rng()
        self._hidden_size = hidden_size
        self._visible_size = visible_size
        self._weights = self._rng.normal(0, 0.1, size=(visible_size, hidden_size)).astype(np.float32)
        self._hidden_bias = self._rng.normal(0, 0.1, size=(hidden_size, 1)).astype(np.float32)
        self._visible_bias = self._rng.normal(0, 0.1, size=(visible_size, 1)).astype(np.float32)
        self._start_temp = start_temp
        self._temp_step = temp_step
        self._learn_rate = learn_rate

    @property
    def visible_size(self):
        return self._visible_size

    def work(self, inputs: npt.NDArray[np.bool], inputs_mask: npt.NDArray[np.bool], num_iterations: int
             ) -> tuple[npt.NDArray[np.bool], npt.NDArray[np.bool]]:
        if inputs.size != inputs_mask.sum():
            raise ValueError("Input size does not match input mask size.")

        outputs_mask = ~inputs_mask
        visible, hidden = self._generate_neurons(inputs, inputs_mask)
        self._simulate_annealing(visible, hidden, outputs_mask, num_iterations=num_iterations)
        return visible, outputs_mask

    def train(self, sample: npt.NDArray[np.bool]):
        visible, hidden = self._generate_neurons(sample, np.ones_like(sample).ravel())
        hidden_energy_delta = (self._weights.T @ visible) + self._hidden_bias
        hidden_prob = _sigmoid(hidden_energy_delta)
        self._simulate_annealing(visible, hidden, np.ones_like(visible), num_iterations=1)
        self._visible_bias = self._visible_bias + (self._learn_rate * (sample.astype(np.int8) - visible))
        self._hidden_bias = self._hidden_bias + (self._learn_rate * (hidden_prob - hidden))
        self._weights = self._weights + (self._learn_rate * ((sample @ hidden_prob.T) - (visible @ hidden.T)))

    def _simulate_annealing(self, visible: npt.NDArray[np.bool], hidden: npt.NDArray[np.bool],
                            outputs_mask: npt.NDArray[np.bool], num_iterations: int):
        temp = self._start_temp
        _ = self._single_pass(visible, hidden, outputs_mask, temp)
        for _ in range(num_iterations):
            _ = self._single_pass(visible, hidden, outputs_mask, temp)
            temp -= max(self._temp_step, 0)

    def _single_pass(self, visible: npt.NDArray[np.bool], hidden: npt.NDArray[np.bool],
                     outputs_mask: npt.NDArray[np.bool], temp: float) -> int:
        hidden_energy_delta = (self._weights.T @ visible) + self._hidden_bias
        hidden_prob = _sigmoid(hidden_energy_delta / temp)
        next_hidden = self._rng.uniform(0, 1, size=hidden.shape) < hidden_prob
        changes = np.sum(hidden != next_hidden)
        hidden[:] = next_hidden

        visible_energy_delta = (self._weights @ hidden) + self._visible_bias
        visible_prob = _sigmoid(visible_energy_delta / temp)
        next_visible = (self._rng.uniform(0, 1, size=visible.shape) < visible_prob)[outputs_mask]
        changes += np.sum(visible[outputs_mask] != next_visible)
        visible[outputs_mask] = next_visible

        return changes

    def _generate_neurons(self, inputs: npt.NDArray[np.bool], inputs_mask: npt.NDArray[np.bool],
                          ) -> tuple[npt.NDArray[np.bool], npt.NDArray[np.bool]]:
        visible = self._rng.integers(0, 2, size=(self._visible_size, 1), dtype=np.bool).astype(np.bool)
        visible[inputs_mask] = inputs
        hidden = self._rng.integers(0, 2, size=(self._hidden_size, 1), dtype=np.bool).astype(np.bool)
        return visible, hidden
