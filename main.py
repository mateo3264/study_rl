from experiments import run_experiment
from environment.environments import EnvWithStimuli, MatchingToSample
from features import *
from scipy.stats import t


if __name__ == '__main__':
    env = MatchingToSample(latency=0, n_stimuli=10)
    features = [get_bias, get_last_action, get_model_stimuli, get_if_last_step]
    features2 = [get_bias, get_last_action, get_model_stimuli, get_if_last_step]
    sample_size = 15
    #samples = run_experiment(env, 2, [features, features2], ['sarsa', 'q-learning'], sample_size=sample_size, n_episodes=15000, verbose=True)
    samples = run_experiment(env, 1, [features], ['dqn'], sample_size=2, n_episodes=20000)

    print('samples:')
    print(samples)
    sarsa_sample = np.array(samples['sarsa0'])
    q_learning_sample = np.array(samples['q-learning1'])
    q_learning_sample
    sarsa_mean = sarsa_sample.mean()
    q_learning_mean = q_learning_sample.mean()

    sarsa_sample_std = sarsa_sample.std(ddof=1)
    q_learning_sample_std = q_learning_sample.std(ddof=1)
    print('sarsa_mean: ', sarsa_mean)
    print('q_learning_mean: ', q_learning_mean)
    print('sarsa_sample_std: ', sarsa_sample_std)
    print('q_learning_sample_std: ', q_learning_sample_std)

    degrees_of_freedom_num = ((sarsa_sample_std**2/sample_size) + (q_learning_sample_std**2/sample_size))**2
    degrees_of_freedom_den = (1/(sample_size - 1))*(sarsa_sample_std**2/sample_size)**2 + (1/(sample_size - 1))*(q_learning_sample_std**2/sample_size)**2

    degrees_of_freedom = degrees_of_freedom_num/degrees_of_freedom_den
    standard_error = np.sqrt(((q_learning_sample_std**2)/sample_size) + ((sarsa_sample_std**2)/sample_size))
    t_value = (q_learning_mean - sarsa_mean)/standard_error
    #df = 9; alpha = 0.001
    alpha = 0.001
    critical_value = t.ppf(q=1-alpha/2,df=degrees_of_freedom)
    print('t_value: ', t_value)
    print(f'degrees of freedom: {degrees_of_freedom}')
    print('critical value: ', critical_value)
    upper_range = q_learning_mean - sarsa_mean + critical_value*standard_error
    lower_range = q_learning_mean - sarsa_mean - critical_value*standard_error
    print('confidence interval: ', lower_range, upper_range)