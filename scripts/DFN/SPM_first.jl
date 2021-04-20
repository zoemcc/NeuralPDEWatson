begin
using DrWatson
import DrWatson.savename
end
@quickactivate "NeuralPDEWatson"
begin
DrWatson.greet()

using Symbolics
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using DifferentialEquations, DiffEqBase
using Quadrature, Cubature, Cuba, QuasiMonteCarlo
using Parameters
using NamedTupleTools
using Plots, LaTeXStrings
using DelimitedFiles
using CSV
using DataFrames
using GLMakie
using Observables
using AbstractPlotting
using StaticArrays
using Hyperopt
using Logging
using Zygote
end



begin
# ('negative particle',) -> rn
# ('positive particle',) -> rp
@parameters t rn rp
# 'Discharge capacity [A.h]' -> Q
# 'X-averaged negative particle concentration' -> c_s_n_xav
# 'X-averaged positive particle concentration' -> c_s_p_xav
@variables Q(..) c_s_n_xav(..) c_s_p_xav(..)
#@variables Q(t) c_s_n_xav(t, rn) c_s_p_xav(t, rp)
Dt = Differential(t)
Drn = Differential(rn)
Drp = Differential(rp)

#=
# 'X-averaged negative particle concentration' equation
cache_4647021298618652029 = 8.813457647415216 * (1 / rn^2 * Drn(rn^2 * Drn(c_s_n_xav(t, rn, rp))))

# 'X-averaged positive particle concentration' equation
cache_m620026786820028969 = 22.598609352346717 * (1 / rp^2 * Drp(rp^2 * Drp(c_s_p_xav(t, rn, rp))))


eqs = [
   Dt(Q(t, rn, rp)) ~ 4.27249308415467,
   Dt(c_s_n_xav(t, rn, rp)) ~ cache_4647021298618652029,# + 1e-7*rp,
   Dt(c_s_p_xav(t, rn, rp)) ~ cache_m620026786820028969,# + 1e-7*rn,
]

ics_bcs = [
   Q(0, rn, rp) ~ 0.0,
   c_s_n_xav(0, rn, rp) ~ 0.8000000000000016,
   c_s_p_xav(0, rn, rp) ~ 0.6000000000000001,
   Drn(c_s_n_xav(t, 0.0, rp)) ~ 0.0,
   Drn(c_s_n_xav(t, 1.0, rp)) ~ -0.14182855923368468,
   Drp(c_s_p_xav(t, rn, 0.0)) ~ 0.0,
   Drp(c_s_p_xav(t, rn, 1.0)) ~ 0.03237700710041634,
]
=#
# 'X-averaged negative particle concentration' equation
cache_4647021298618652029 = 8.813457647415216 * (1 / rn^2 * Drn(rn^2 * Drn(c_s_n_xav(t, rn))))

# 'X-averaged positive particle concentration' equation
cache_m620026786820028969 = 22.598609352346717 * (1 / rp^2 * Drp(rp^2 * Drp(c_s_p_xav(t, rp))))


eqs = [
   Dt(Q(t)) ~ 4.27249308415467,
   Dt(c_s_n_xav(t, rn)) ~ cache_4647021298618652029,
   Dt(c_s_p_xav(t, rp)) ~ cache_m620026786820028969,
]

ics_bcs = [
   Q(0) ~ 0.0,
   c_s_n_xav(0, rn) ~ 0.8000000000000016,
   c_s_p_xav(0, rp) ~ 0.6000000000000001,
   Drn(c_s_n_xav(t, 0.0)) ~ 0.0,
   Drn(c_s_n_xav(t, 1.0)) ~ -0.14182855923368468,
   Drp(c_s_p_xav(t, 0.0)) ~ 0.0,
   Drp(c_s_p_xav(t, 1.0)) ~ 0.03237700710041634,
]

t_domain = IntervalDomain(0.0, 3600.0)
rn_domain = IntervalDomain(0.0, 1.0)
rp_domain = IntervalDomain(0.0, 1.0)

domains = [
   t in t_domain,
   rn in rn_domain,
   rp in rp_domain,
]
ind_vars = [t, rn, rp]
dep_vars = [Q(t), c_s_n_xav(t, rn), c_s_p_xav(t, rp)]

#depvar_int, indvar_int, indvar_dict, depvar_dict = NeuralPDE.get_vars(ind_vars, dep_vars)
#indvar_dict_int = Dict([(indvar, i) for (i, indvar) in enumerate(ind_vars)])
#=
# 'X-averaged negative particle concentration' equation
cache_4647021298618652029 = 8.813457647415216 * (1 / rn^2 * Drn(rn^2 * Drn(c_s_n_xav(t, rn))))

# 'X-averaged positive particle concentration' equation
cache_m620026786820028969 = 22.598609352346717 * (1 / rp^2 * Drp(rp^2 * Drp(c_s_p_xav(t, rp))))


eqs = [
   Dt(Q(t)) ~ 4.27249308415467,
   Dt(c_s_n_xav(t, rn)) ~ cache_4647021298618652029,
   Dt(c_s_p_xav(t, rp)) ~ cache_m620026786820028969,
]

ics_bcs = [
   Q(0) ~ 0.0,
   c_s_n_xav(0, rn) ~ 0.8000000000000016,
   c_s_p_xav(0, rp) ~ 0.6000000000000001,
   Drn(c_s_n_xav(t, 0.0)) ~ 0.0,
   Drn(c_s_n_xav(t, 1.0)) ~ -0.14182855923368468,
   Drp(c_s_p_xav(t, 0.0)) ~ 0.0,
   Drp(c_s_p_xav(t, 1.0)) ~ 0.03237700710041634,
]

t_domain = IntervalDomain(0.0, 3600.0)
rn_domain = IntervalDomain(0.0, 1.0)
rp_domain = IntervalDomain(0.0, 1.0)

domains = [
   t in t_domain,
   rn in rn_domain,
   rp in rp_domain,
]
#ind_vars = [t, rn, rp]
ind_vars = [t, rn]
dep_vars = [Q, c_s_n_xav]
=#

SPM_pde_system = PDESystem(eqs, ics_bcs, domains, ind_vars, dep_vars)
end



## PINN Part


begin
num_dim = 50
nonlin = Flux.gelu
#in_dim = 3
#out_dim = 3
#strategy_ = NeuralPDE.QuadratureTraining(;quadrature_alg=HCubatureJL(),abstol=1e-6, reltol=1e-8, maxiters=4000, batch=0)
#strategy_ = NeuralPDE.QuadratureTraining(;abstol=1e-6, reltol=1e-8, maxiters=2000)
#strategy_ = NeuralPDE.QuadratureTraining(;)
strategy_ = NeuralPDE.StochasticTraining(128)
#in_dims = [3, 3, 3]
#in_dims = [1, 2, 2]
in_dims = [1, 2, 2]
num_hid = 3
chains_ = [FastChain(FastDense(in_dim,num_dim,nonlin),
                     [FastDense(num_dim,num_dim,nonlin) for i in 1:num_hid]...,
                     FastDense(num_dim,1)) for in_dim in in_dims]
#adalosspoisson = NeuralPDE.LossGradientsAdaptiveLoss(20; α=0.9f0)
adalosspoisson = NeuralPDE.NonAdaptiveLossWeights()
discretization = NeuralPDE.PhysicsInformedNN(chains_,
                                                strategy_,
                                                adaptive_loss=adalosspoisson)
end

#@run sym_prob = NeuralPDE.symbolic_discretize(SPM_pde_system,discretization)
#@run sym_prob = NeuralPDE.symbolic_discretize(SPM_pde_system,discretization)
sym_prob = NeuralPDE.symbolic_discretize(SPM_pde_system,discretization)
prob = NeuralPDE.discretize(SPM_pde_system,discretization)
begin
initθ = discretization.init_params
initθ = cat(initθ..., dims=1)
opt = Flux.Optimiser(ClipValue(1e-3), ExpDecay(1, 0.5, 25_000), ADAM(3e-4))
saveevery = 100
loss = Float64[]

experiment_path = datadir("stochastic_128_first")
iteration_count_arr = [0]
cb = function (p,l)
    iteration_count = iteration_count_arr[1]

    println("Current loss is: $l, iteration is: $(iteration_count)")
    push!(loss, l)
    if iteration_count % saveevery == 0
        cursavefile = joinpath(experiment_path, string(iteration_count, base=10, pad=5) * ".csv")
        writedlm(cursavefile, p, ",")
        losssavefile = joinpath(experiment_path, "loss.csv")
        writedlm(losssavefile, loss, ",")
    end
    iteration_count_arr[1] += 1
    return false
end
end
#prob.f(initθ, [])
#@run prob.f(initθ, [])
#@run res = GalacticOptim.solve(prob, opt; cb = cb, maxiters=100)
res = GalacticOptim.solve(prob, opt; cb = cb, maxiters=200_000)
phi = discretization.phi