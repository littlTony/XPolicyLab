"""CPU regression for native stats weighting, terminal padding and variance."""
import unittest
import numpy as np
from prepare_stats import native_statistics, ACTION_HORIZON


class NativeStatsTests(unittest.TestCase):
    def test_unequal_episodes_native_reducer_semantics(self):
        values = [np.array([0., 2.], np.float32),
                  np.array([10., 12., 14., 16.], np.float32)]
        episodes = [(np.repeat(x[:, None], 58, axis=1),
                     np.repeat(x[:, None], 58, axis=1)) for x in values]
        result = native_statistics(episodes)
        self.assertEqual(result['num_episodes'], 2)
        self.assertEqual(result['num_transition'], 6)
        for section in ['state', 'action']:
            means, variances, minima, maxima = [], [], [], []
            q01s, q99s = [], []
            for x in values:
                offsets = [0] if section == 'state' else range(ACTION_HORIZON)
                # Python lists deliberately construct the expected windows
                # independently of the native torch sliding-window helper.
                samples = np.array([[float(x[min(t + k, len(x) - 1)])
                                     for k in offsets] for t in range(len(x))])
                means.append(samples.mean(axis=0))
                variances.append(samples.var(axis=0, ddof=1))
                minima.append(samples.min(axis=0)); maxima.append(samples.max(axis=0))
                q01s.append(np.quantile(samples, 0.01, axis=0))
                q99s.append(np.quantile(samples, 0.99, axis=0))
            means = np.stack(means); variances = np.stack(variances)
            global_mean = means.mean()
            expected = {
                'global_mean': global_mean,
                'global_std': np.sqrt((variances + (means-global_mean)**2).mean()),
                'global_min': np.min(minima), 'global_max': np.max(maxima),
                'global_q01': np.min(q01s), 'global_q99': np.max(q99s),
                'stepwise_mean': means.mean(0),
                'stepwise_std': np.sqrt((variances+(means-means.mean(0))**2).mean(0)),
                'stepwise_min': np.min(minima, axis=0),
                'stepwise_max': np.max(maxima, axis=0),
                'stepwise_q01': np.min(q01s, axis=0),
                'stepwise_q99': np.max(q99s, axis=0),
            }
            actual = result[section]['default']
            for key, value in expected.items():
                tensor = actual[key].numpy()
                value = np.broadcast_to(np.asarray(value)[..., None], tensor.shape)
                np.testing.assert_allclose(tensor, value, rtol=2e-6, atol=2e-6,
                                           err_msg=f'{section}.{key}')
        self.assertAlmostEqual(float(result['state']['default']['global_mean'][0]), 7.)
        self.assertNotAlmostEqual(float(result['action']['default']['global_mean'][0]), 7.)
        self.assertNotAlmostEqual(float(result['state']['default']['global_mean'][0]),
                                  float(np.concatenate(values).mean()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
