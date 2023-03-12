from features import *
from agents import linear_agents
from environment.environments import EnvWithStimuli, MatchingToSample


features = [get_bias, get_last_action, get_model_stimuli, get_if_last_step]
learn = Learn('sarsa')


env = MatchingToSample()
feta = linear_agents.FeatEligibilityTracesAgent(env, learn, features)
state = np.array([[0, 0],[0, 1]])

feta.update_w(state, 0, state, 1, False)