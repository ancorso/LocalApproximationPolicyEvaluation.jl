using LocalApproximationPolicyEvaluation
using Test
using POMDPModels
using POMDPs
using GridInterpolations
using LocalFunctionApproximation

# Step 1 - Setup the gridworld problem
g_size = (9,9)
g = SimpleGridWorld(size = g_size, rewards = Dict(GWPos(g_size...) => 1, GWPos(1,1) => 0), tprob = 1., discount=1)
action_probability(g::SimpleGridWorld, s, a) = 0.25
POMDPs.convert_s(::Type{AbstractArray}, s::GWPos, g::SimpleGridWorld) = SVector{4, Float64}(1., s[1], s[2], s[1]*s[2])
POMDPs.gen(mdp::SimpleGridWorld, s, a, rng = Random.GLOBAL_RNG) = (sp =rand(transition(g, s, a )), r=reward(g, s, a))
POMDPs.initialstate(g::SimpleGridWorld) = initialstate_distribution(g)



# Step 2 - Solve the problem semi-exactly using local approximation
grid = RectangleGrid([1:g_size[1] ...], [1:g_size[2]...])
interp = LocalGIFunctionApproximator(grid)
solver = LocalPolicyEvalSolver(interp, action_probability, is_mdp_generative = true, n_generative_samples = 1, max_iterations = 2000, belres = 1e-6)
policy = solve(solver, g)
values = [value(policy, s) for s in states(g)]
@test value(policy, GWPos(9,9)) == 1.0
@test value(policy, GWPos(8,9)) > .25
@test value(policy, GWPos(1,1)) == 0

