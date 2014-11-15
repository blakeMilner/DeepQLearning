require 'xlua'
local Brain = require 'deepqlearn'

function randtable(size, startnum, endnum) 
	local rtable = {}
	for i = 1, size do
		rtable[i+1] = randf(startnum, endnum)
	end
	
  return rtable
end

-- simple test found in readme.md
num_outcomes = 3


Brain.init(num_outcomes, num_outcomes)   
nb_train = 1000
nb_test  = 1000

for k = 0, nb_train do
	rand_outcome = math.random(1, num_outcomes)
	state = randtable(num_outcomes, rand_outcome, rand_outcome + 1)
	
   xlua.progress(k, nb_train)
   
   newstate = table.copy(state) -- make a deep copy
   action = Brain.forward(newstate); -- returns index of chosen action
    
   reward = (action == rand_outcome) and 1 or 0
   
   Brain.backward(reward); -- learning magic happens 
end

Brain.epsilon_test_time = 0.0; -- don't make any more random choices
Brain.learning = false;


-- get an optimal action from the learned policy
local cnt = 0
for k = 1, nb_test do
	xlua.progress(k, nb_test)
   
	rand_outcome = math.random(1, num_outcomes)
	state = randtable(num_outcomes, rand_outcome, rand_outcome + 1)
	
  
   newstate = table.copy(state)
   output = Brain.forward(newstate)
   if rand_outcome == output then
      cnt = cnt + 1   	
   end
   
end

print("Test cases correct: " .. tostring(100 * cnt/nb_test) .. " %")

