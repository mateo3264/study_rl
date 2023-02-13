from experiments import run_experiment
from environment.environments import EnvWithStimuli, MatchingToSample
from features import *


if __name__ == '__main__':
    env = MatchingToSample(latency=20, n_stimuli=10)
    features = [get_bias, get_model_stimuli, get_if_last_step]
    features2 = [get_bias, get_last_action, get_model_stimuli, get_if_last_step]
    samples = run_experiment(env, 2, [features, features2], ['sarsa', 'sarsa'], sample_size=3, n_episodes=5000, verbose=True)
    #samples = run_experiment(env, 1, [features], ['q-learning'], sample_size=2, n_episodes=10000)

    print('samples:')
    print(samples)
    sarsa_sample = np.array(samples['sarsa0'])
    q_learning_sample = np.array(samples['sarsa1'])
    q_learning_sample
    sarsa_mean = sarsa_sample.mean()
    q_learning_mean = q_learning_sample.mean()

    sarsa_sample_std = sarsa_sample.std(ddof=1)
    q_learning_sample_std = q_learning_sample.std(ddof=1)
    print('sarsa_mean: ', sarsa_mean)
    print('q_learning_mean: ', q_learning_mean)
    print('sarsa_sample_std: ', sarsa_sample_std)
    print('q_learning_sample_std: ', q_learning_sample_std)


    standard_error = np.sqrt(((q_learning_sample_std**2)/len(q_learning_sample)) + ((sarsa_sample_std**2)/len(sarsa_sample)))
    t_value = (q_learning_mean - sarsa_mean)/standard_error
    print('t_value: ', t_value)
    print('critical value: ', 2.262)
    upper_range = q_learning_mean - sarsa_mean + 2.262*standard_error
    lower_range = q_learning_mean - sarsa_mean - 2.262*standard_error
    print('confidence interval: ', lower_range, upper_range)