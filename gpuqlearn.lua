require('math')
require('nnx')
require('os')
require('optim')
require('cutorch')
require('cunn')
math.randomseed(os.time())
torch.setdefaulttensortype('torch.FloatTensor')
local Brain = { }
randf = function(s, e)
  return (math.random(0, (e - s) * 9999) / 10000) + s
end
table.merge = function(t1, t2)
  local t = t1
  for i = 1, #t2 do
    t[#t + 1] = t2[i]
  end
  return t
end
table.copy = function(t)
  local u
  do
    local _tbl_0 = { }
    for k, v in pairs(t) do
      _tbl_0[k] = v
    end
    u = _tbl_0
  end
  return setmetatable(u, getmetatable(t))
end
table.length = function(T)
  local count = 0
  for _ in pairs(T) do
    count = count + 1
  end
  return count
end
Experience = function(state0, action0, reward0, state1)
  local NewExperience = {
    state0 = state0,
    action0 = action0,
    reward0 = reward0,
    state1 = state1
  }
  return NewExperience
end
Brain.init = function(num_states, num_actions)
  Brain.temporal_window = 2
  Brain.experience_size = 30000
  Brain.start_learn_threshold = 300
  Brain.gamma = 0.9
  Brain.learning_steps_total = 100000
  Brain.learning_steps_burnin = 300
  Brain.epsilon = 1.0
  Brain.epsilon_min = 0.05
  Brain.epsilon_test_time = 0.01
  local _ = [[== states and actions that go into neural net:
		(state0,action0),(state1,action1), ... , (stateN)
		this variable controls the size of that temporal window.
	]]
  Brain.net_inputs = (num_states + num_actions) * Brain.temporal_window + num_states
  Brain.hidden_nodes = 16
  Brain.num_states = num_states
  Brain.num_actions = num_actions
  Brain.net_outputs = Brain.num_actions
  _ = [[== Window size dictates the number of states, actions, rewards, and net inputs that we
		save. The temporal window size is the number of time states/actions that are input
		to the network and must be smaller than or equal to window_size
	]]
  Brain.window_size = math.max(Brain.temporal_window, 2)
  Brain.random_action_distribution = { }
  if table.length(Brain.random_action_distribution) > 0 then
    if table.length(Brain.random_action_distribution) ~= Brain.num_actions then
      print('TROUBLE. random_action_distribution should be same length as num_actions.')
    end
    local s = 0.0
    for k = 1, table.length(Brain.random_action_distribution) do
      s = s + Brain.random_action_distribution[k]
    end
    if math.abs(s - 1.0) > 0.0001 then
      print('TROUBLE. random_action_distribution should sum to 1!')
    end
  end
  Brain.net = nn.Sequential()
  Brain.net:add(nn.Linear(Brain.net_inputs, Brain.hidden_nodes))
  Brain.net:add(nn.Threshold(0, 0))
  Brain.net:add(nn.Linear(Brain.hidden_nodes, Brain.hidden_nodes))
  Brain.net:add(nn.Threshold(0, 0))
  Brain.net:add(nn.Linear(Brain.hidden_nodes, Brain.net_outputs))
  Brain.net:cuda()
  Brain.criterion = nn.MSECriterion():cuda()
  Brain.learning_rate = 0.01
  Brain.learning_rate_decay = 5e-7
  Brain.batch_size = 16
  Brain.momentum = 0.9
  Brain.age = 0
  Brain.forward_passes = 0
  Brain.learning = true
  Brain.coefL1 = 0.001
  Brain.coefL2 = 0.001
  Brain.parameters, Brain.gradParameters = Brain.net:getParameters()
  Brain.experience = { }
  Brain.state_window = { }
  Brain.action_window = { }
  Brain.reward_window = { }
  Brain.net_window = { }
  for i = 1, Brain.window_size do
    Brain.state_window[i] = { }
    Brain.action_window[i] = { }
    Brain.reward_window[i] = { }
    Brain.net_window[i] = { }
  end
end
Brain.random_action = function()
  if table.length(Brain.random_action_distribution) == 0 then
    return (torch.random() % Brain.net_outputs) + 1
  else
    local p = randf(0, 1)
    local cumprob = 0.0
    for k = 1, Brain.num_actions do
      cumprob = cumprob + Brain.random_action_distribution[k]
      if p < cumprob then
        return k
      end
    end
  end
end
Brain.policy = function(state)
  local tensor_state = torch.Tensor(state):cuda()
  local action_values = Brain.net:forward(tensor_state)
  local maxval = action_values[1]
  local max_index = 1
  for i = 2, Brain.net_outputs do
    if action_values[i] > maxval then
      maxval = action_values[i]
      max_index = i
    end
  end
  return {
    action = max_index,
    value = maxval
  }
end
Brain.getNetInput = function(xt)
  local w = { }
  w = table.merge(w, xt)
  local n = Brain.window_size + 1
  for k = 1, Brain.temporal_window do
    w = table.merge(w, Brain.state_window[n - k])
    local action1ofk = { }
    for i = 1, Brain.num_actions do
      action1ofk[i] = 0
    end
    action1ofk[Brain.action_window[n - k]] = 1.0 * Brain.num_states
    w = table.merge(w, action1ofk)
  end
  return w
end
Brain.forward = function(input_array)
  Brain.forward_passes = Brain.forward_passes + 1
  local action, net_input
  if Brain.forward_passes > Brain.temporal_window then
    net_input = Brain.getNetInput(input_array)
    if Brain.learning then
      local new_epsilon = 1.0 - (Brain.age - Brain.learning_steps_burnin) / (Brain.learning_steps_total - Brain.learning_steps_burnin)
      Brain.epsilon = math.min(1.0, math.max(Brain.epsilon_min, new_epsilon))
    else
      Brain.epsilon = Brain.epsilon_test_time
    end
    if randf(0, 1) < Brain.epsilon then
      action = Brain.random_action()
    else
      local best_action = Brain.policy(net_input)
      action = best_action.action
    end
  else
    net_input = { }
    action = Brain.random_action()
  end
  table.remove(Brain.net_window, 1)
  table.insert(Brain.net_window, net_input)
  table.remove(Brain.state_window, 1)
  table.insert(Brain.state_window, input_array)
  table.remove(Brain.action_window, 1)
  table.insert(Brain.action_window, action)
  return action
end
Brain.backward = function(reward)
  table.remove(Brain.reward_window, 1)
  table.insert(Brain.reward_window, reward)
  if not (Brain.learning) then
    return 
  end
  Brain.age = Brain.age + 1
  if Brain.forward_passes > Brain.temporal_window + 1 then
    local e = Experience(nil, nil, nil, nil)
    local n = Brain.window_size
    e.state0 = Brain.net_window[n - 1]
    e.action0 = Brain.action_window[n - 1]
    e.reward0 = Brain.reward_window[n - 1]
    e.state1 = Brain.net_window[n]
    if table.length(Brain.experience) < Brain.experience_size then
      table.insert(Brain.experience, e)
    else
      local ri = torch.random(1, Brain.experience_size)
      Brain.experience[ri] = e
    end
  end
  if table.length(Brain.experience) > Brain.start_learn_threshold then
    local inputs = torch.Tensor(Brain.batch_size, Brain.net_inputs):cuda()
    local targets = torch.Tensor(Brain.batch_size, Brain.net_outputs):cuda()
    for k = 1, Brain.batch_size do
      local re = math.random(1, table.length(Brain.experience))
      local e = Brain.experience[re]
      local x = torch.Tensor(e.state0):cuda()
      local best_action = Brain.policy(e.state1)
      local all_outputs = Brain.net:forward(x)
      inputs[k] = x:clone()
      targets[k] = all_outputs:clone()
      targets[k][e.action0] = e.reward0 + Brain.gamma * best_action.value
    end
    local feval
    feval = function(x)
      collectgarbage()
      if not (x == Brain.parameters) then
        Brain.parameters:copy(x)
      end
      Brain.gradParameters:zero()
      local outputs = Brain.net:forward(inputs)
      local f = Brain.criterion:forward(outputs, targets)
      local df_do = Brain.criterion:backward(outputs, targets)
      Brain.net:backward(inputs, df_do)
      if Brain.coefL1 ~= 0 or Brain.coefL2 ~= 0 then
        local norm, sign = torch.norm, torch.sign
        f = f + (Brain.coefL1 * norm(Brain.parameters, 1))
        f = f + (Brain.coefL2 * 0.5 * norm(Brain.parameters, 2) ^ 2)
        Brain.gradParameters:add(sign(Brain.parameters):mul(Brain.coefL1) + Brain.parameters:clone():mul(Brain.coefL2))
      end
      return f, Brain.gradParameters
    end
    local sgdState = {
      learningRate = Brain.learning_rate,
      momentum = Brain.momentum,
      learningRateDecay = Brain.learning_rate_decay
    }
    return optim.sgd(feval, Brain.parameters, sgdState)
  end
end
return Brain
