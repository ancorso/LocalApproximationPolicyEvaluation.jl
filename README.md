# LocalApproximationPolicyEvaluation.jl
A local approximation dynamic programming approach to policy evaluation. This algorithm computes the expected return at each state given a probability distribution over actions.

This code is adapted from: https://github.com/JuliaPOMDP/LocalApproximationValueIteration.jl

## Usage Notes
* Requires a generative MDP with a discrete action space
* Requires a function that computes the log-probability of each action
* Returns a policy that
  1. Can have an action sampled according to the expected return of each action
  2. Can have the value of the state or state-action computed
* See `test/gridworld_test.jl` for a usage example on a gridworld problem.


Maintained by Anthony Corso (acorso@stanford.edu)
