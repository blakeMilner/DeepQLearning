require 'math'
require 'nnx'
require 'os'
require 'optim'
require 'cutorch'
require 'cunn'

math.randomseed os.time!
torch.setdefaulttensortype 'torch.FloatTensor'

Brain = {}

--  HELPER FUNCTIONS --
 
export randf = (s, e) ->
	return (math.random(0, (e - s) * 9999) / 10000) + s

-- new methods for table

table.merge = (t1, t2) ->
	t = t1
	for i = 1, #t2
		t[#t + 1] = t2[i]
	return t

table.copy = (t) ->
	u = {k, v for k, v in pairs t}
	return setmetatable(u, getmetatable t)

table.length = (T) ->
	count = 0
	count += 1 for _ in pairs T
	return count

-- returns experience table for single network decision
-- contains the state, action chosen, whether a reward was obtained, and the
-- state that resulted from the action. This is later used to train the network
-- Remember that the utility of an action is evaluated from the reward gained and
-- the utility of the state it led to (recursive definition)
export Experience = (state0, action0, reward0, state1) ->
	NewExperience =
		state0: state0
		action0: action0
		reward0: reward0
		state1: state1
	return NewExperience

-- BRAIN

Brain.init = (num_states, num_actions) ->
	-- Number of past state/action pairs input to the network. 0 = agent lives in-the-moment :)
	Brain.temporal_window = 2
	-- Maximum number of experiences that we will save for training
	Brain.experience_size = 30000
	-- experience necessary to start learning
	Brain.start_learn_threshold = 300
	-- gamma is a crucial parameter that controls how much plan-ahead the agent does. In [0,1]
	-- Determines the amount of weight placed on the utility of the state resulting from an action.
	Brain.gamma = 0.9
	-- number of steps we will learn for
	Brain.learning_steps_total = 100000
	-- how many steps of the above to perform only random actions (in the beginning)?
	Brain.learning_steps_burnin = 300
	-- controls exploration exploitation tradeoff. Will decay over time
	-- a higher epsilon means we are more likely to choose random actions
	Brain.epsilon = 1.0
	-- what epsilon value do we bottom out on? 0.0 => purely deterministic policy at end
	Brain.epsilon_min = 0.05
	-- what epsilon to use when learning is turned off. This is for testing
	Brain.epsilon_test_time = 0.01

	[[== states and actions that go into neural net:
		(state0,action0),(state1,action1), ... , (stateN)
		this variable controls the size of that temporal window.
	]]
	Brain.net_inputs = (num_states + num_actions) * Brain.temporal_window + num_states
	Brain.hidden_nodes = 16
	Brain.num_states = num_states
	Brain.num_actions = num_actions
	Brain.net_outputs = Brain.num_actions

	[[== Window size dictates the number of states, actions, rewards, and net inputs that we
		save. The temporal window size is the number of time states/actions that are input
		to the network and must be smaller than or equal to window_size
	]]
	Brain.window_size = math.max Brain.temporal_window, 2

	-- advanced feature. Sometimes a random action should be biased towards some values
	-- for example in flappy bird, we may want to choose to not flap more often
	Brain.random_action_distribution = {}
	if table.length(Brain.random_action_distribution) > 0
		-- this better sum to 1 by the way, and be of length this.num_actions
		if table.length(Brain.random_action_distribution) != Brain.num_actions
			print 'TROUBLE. random_action_distribution should be same length as num_actions.'
		
		s = 0.0
		
		for k = 1, table.length Brain.random_action_distribution
			s += Brain.random_action_distribution[k]
		
		if math.abs(s - 1.0) > 0.0001
			 print 'TROUBLE. random_action_distribution should sum to 1!'


	-- define architecture
	Brain.net = nn.Sequential!

	Brain.net\add nn.Linear Brain.net_inputs, Brain.hidden_nodes
	Brain.net\add nn.Threshold 0, 0

	Brain.net\add nn.Linear Brain.hidden_nodes, Brain.hidden_nodes
	Brain.net\add nn.Threshold 0, 0

	Brain.net\add nn.Linear Brain.hidden_nodes, Brain.net_outputs

	Brain.net\cuda! -- move network to GPU

	Brain.criterion = nn.MSECriterion!\cuda!


	-- other learning parameters
	Brain.learning_rate = 0.01
	Brain.learning_rate_decay = 5e-7
	Brain.batch_size = 16
	Brain.momentum = 0.9
		
	-- various housekeeping variables
	Brain.age = 0 -- incremented every backward!

	-- number of times we've called forward - lets us know when our input temporal
	-- window is filled up
	Brain.forward_passes = 0
	Brain.learning = true

	-- coefficients for regression
	Brain.coefL1 = 0.001
	Brain.coefL2 = 0.001

	-- parameters for optim.sgd
	Brain.parameters, Brain.gradParameters = Brain.net\getParameters!

	-- These windows track old experiences, states, actions, rewards, and net inputs
	-- over time. They should all start out as empty with a fixed size.
	-- This is a first in, last out data structure that is shifted along time
	Brain.experience = {}
	Brain.state_window = {}
	Brain.action_window = {}
	Brain.reward_window = {}
	Brain.net_window = {}
	for i = 1, Brain.window_size
		Brain.state_window[i] = {}
		Brain.action_window[i] = {}
		Brain.reward_window[i] = {}
		Brain.net_window[i] = {}

-- a bit of a helper function. It returns a random action
-- we are abstracting this away because in future we may want to
-- do more sophisticated things. For example some actions could be more
-- or less likely at "rest"/default state.
Brain.random_action = ->
	-- if we don't have a random action distribution defined then sample evenly
	if table.length(Brain.random_action_distribution) == 0
		return (torch.random! % Brain.net_outputs) + 1

	-- okay, lets do some fancier sampling:
	else
		p = randf 0, 1
		cumprob = 0.0

		for k = 1, Brain.num_actions
			cumprob += Brain.random_action_distribution[k]
			
			if p < cumprob
				return k

-- compute the value of doing any action in this state
-- and return the argmax action and its value
Brain.policy = (state) ->
	tensor_state = torch.Tensor(state)\cuda!
	action_values = Brain.net\forward tensor_state
	
	maxval = action_values[1]
	max_index = 1
 
	-- find maximum output and note its index and value
	--max_index = i for i = 2, Brain.net_outputs when action_values[i] > action_values[max_index]
	for i = 2, Brain.net_outputs
		if action_values[i] > maxval
			maxval = action_values[i]
			max_index = i
	
	return action: max_index, value: maxval
		
-- This function assembles the input to the network by concatenating
-- old (state, chosen_action) pairs along with the current state
	-- return s = (x,a,x,a,x,a,xt) state vector.
Brain.getNetInput = (xt) ->
	w = {}
	w = table.merge(w, xt) -- start with current state
	
	-- and now go backwards and append states and actions from history temporal_window times
	n = Brain.window_size + 1
	for k = 1, Brain.temporal_window do
		-- state
		w = table.merge w, Brain.state_window[n - k]
		-- action, encoded as 1-of-k indicator vector. We scale it up a bit because
		-- we don't want weight regularization to undervalue this information, as it only exists once
		action1ofk = {}
		action1ofk[i] = 0 for i = 1, Brain.num_actions

		-- assign action taken for current state to be 1, all others are 0
		action1ofk[Brain.action_window[n - k]] = 1.0 * Brain.num_states
			
		w = table.merge w, action1ofk
	
	return w
		
-- This function computes an action by either:
-- 1. Giving the current state and past (state, action) pairs to the network
-- and letting it choose the best acction
-- 2. Choosing a random action
Brain.forward = (input_array) ->
	Brain.forward_passes += 1
	
	local action, net_input
	
	-- if we have enough (state, action) pairs in our memory to fill up
	-- our network input then we'll proceed to let our network choose the action
	if Brain.forward_passes > Brain.temporal_window
		net_input = Brain.getNetInput input_array
		
		-- if learning is turned on then epsilon should be decaying
		if Brain.learning
			-- compute (decaying) epsilon for the epsilon-greedy policy
			new_epsilon = 1.0 - (Brain.age - Brain.learning_steps_burnin)/(Brain.learning_steps_total - Brain.learning_steps_burnin)
			
			-- don't let epsilon go above 1.0
			Brain.epsilon = math.min(1.0, math.max(Brain.epsilon_min, new_epsilon))
		else
			-- if learning is turned off then use the epsilon we've specified for testing
			Brain.epsilon = Brain.epsilon_test_time
		
		-- use epsilon probability to choose whether we use network action or random action
		if randf(0, 1) < Brain.epsilon
			action = Brain.random_action!
		else
			-- otherwise use our policy to make decision
			best_action = Brain.policy net_input
			action = best_action.action -- this is the action number
	else
		-- pathological case that happens first few iterations when we can't
		-- fill up our network inputs. Just default to random action in this case
		net_input = {}
		action = Brain.random_action!
	
	-- shift the network input, state, and action chosen into our windows
	table.remove Brain.net_window, 1
	table.insert Brain.net_window, net_input

	table.remove Brain.state_window, 1
	table.insert Brain.state_window, input_array

	table.remove Brain.action_window, 1
	table.insert Brain.action_window, action
	
	return action
		
-- This function trains the network using the reward resulting from the last action
-- It will save this past experience which consists of:
--  the state, action chosen, whether a reward was obtained, and the
--  state that resulted from the action
-- After that, it will train the network (using a batch of experiences) using a
-- random sampling of our entire experience history.
Brain.backward = (reward) ->
	-- add reward to our history
	table.remove Brain.reward_window, 1
	table.insert Brain.reward_window, reward
	
	-- if learning is turned off then don't do anything
	return unless Brain.learning
	
	Brain.age += 1
	
	-- if we've had enough states and actions to fill up our net input then add
	-- this new experience to our history
	if Brain.forward_passes > Brain.temporal_window + 1
		-- make experience and fill it up
		e = Experience nil, nil, nil, nil
		n = Brain.window_size
		e.state0 = Brain.net_window[n - 1]
		e.action0 = Brain.action_window[n - 1]
		e.reward0 = Brain.reward_window[n - 1]
		e.state1 = Brain.net_window[n]
		
		-- if our experience table isn't larger than the max size then expand
		if table.length(Brain.experience) < Brain.experience_size
			table.insert Brain.experience, e
		-- Otherwise replace random experience. finite memory!
		else
			ri = torch.random 1, Brain.experience_size
			Brain.experience[ri] = e
	
	-- if we have enough experience in memory then start training
	if table.length(Brain.experience) > Brain.start_learn_threshold
		inputs = torch.Tensor(Brain.batch_size, Brain.net_inputs)\cuda!
		targets = torch.Tensor(Brain.batch_size, Brain.net_outputs)\cuda!

		for k = 1, Brain.batch_size
			-- choose random experience
			re = math.random 1, table.length Brain.experience
			e = Brain.experience[re]
			
			-- copy state from experience
			x = torch.Tensor(e.state0)\cuda!

			-- compute best action for the new state
			best_action = Brain.policy e.state1

			-- get current action output values
			-- we want to make the target outputs the same as the actual outputs
			-- expect for the action that was chose - we want to replace this with
			-- the reward that was obtained + the utility of the resulting state
			all_outputs = Brain.net\forward x
			inputs[k] = x\clone!
			targets[k] = all_outputs\clone!
			targets[k][e.action0] = e.reward0 + Brain.gamma * best_action.value

		-- create training function to give to optim.sgd
		feval = (x) ->
			collectgarbage!

			-- get new network parameters
			Brain.parameters\copy x unless x == Brain.parameters

			-- reset gradients
			Brain.gradParameters\zero!

			-- evaluate function for complete mini batch
			outputs = Brain.net\forward inputs
			f = Brain.criterion\forward outputs, targets

			-- estimate df/dW
			df_do = Brain.criterion\backward outputs, targets
			Brain.net\backward inputs, df_do

			-- penalties (L1 and L2):
			if Brain.coefL1 != 0 or Brain.coefL2 != 0
				-- locals:
				norm,sign = torch.norm, torch.sign

				-- Loss:
				f += Brain.coefL1 * norm Brain.parameters, 1
				f += Brain.coefL2 * 0.5 * norm(Brain.parameters, 2) ^ 2

				-- Gradients:
				Brain.gradParameters\add(sign(Brain.parameters)\mul(Brain.coefL1) + Brain.parameters\clone!\mul Brain.coefL2)

			-- return f and df/dX
			return f, Brain.gradParameters

		-- fire up optim.sgd
		sgdState =
			learningRate: Brain.learning_rate
			momentum: Brain.momentum
			learningRateDecay: Brain.learning_rate_decay
		
		optim.sgd feval, Brain.parameters, sgdState



-- export
return Brain