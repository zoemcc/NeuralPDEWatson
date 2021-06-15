#begin
#using DrWatson
#import DrWatson.savename
#end
#@quickactivate "NeuralPDEWatson"
begin
push!(LOAD_PATH, "/home/zobot/.julia/dev/NeuralPDE.jl/src")
using Revise
using LinearAlgebra
using IfElse
using PyCall
using Flux
println("NNPDE_tests_heterogeneous")
using DiffEqFlux
println("Starting Soon!")
using ModelingToolkit
using DiffEqBase
using Test, NeuralPDE
using GalacticOptim
using Optim
using Quadrature,Cubature, Cuba
using QuasiMonteCarlo
using SciMLBase
using DelimitedFiles
using CSV
using OrdinaryDiffEq
using Plots
using DataFrames
using LineSearches
using Zygote

using Random
end

begin
@pyimport pickle
@pyimport numpy

function myunpickle(filename)
    r = nothing
    @pywith pybuiltin("open")(filename,"rb") as f begin
        r = pickle.load(f)
    end
    return r
end
# 'Electrolyte concentration' equation

function concatenation(x, n, s, p)
# A concatenation in the electrolyte domain
IfElse.ifelse(
    x < 0.4444444444444445, n, IfElse.ifelse(
        x < 0.5555555555555556, s, p
    )
)
end
@register concatenation(x, n, s, p)
end

#function main_train()
begin
    #pybamm_sols = myunpickle("/home/zobot/.julia/dev/DFN.jl/pybamm/hardcoded_models/MTK_format/pybamm_solutions/SPM.pickle")

    #sols_t = pybamm_sols["Time"]
    #sols_r_n = pybamm_sols["r_n"][:,1]
    #sols_r_p = pybamm_sols["r_p"][:,1]
    #sols_c_s_n = pybamm_sols["X-averaged negative particle concentration"]
    #sols_c_s_p = pybamm_sols["X-averaged positive particle concentration"]
    begin
    # ('negative particle',) -> rn
    # ('positive particle',) -> rp
    # ('negative electrode', 'separator', 'positive electrode') -> x
    @parameters t rn rp x
    # 'Discharge capacity [A.h]' -> Q
    # 'X-averaged negative particle concentration' -> c_s_n_xav
    # 'X-averaged positive particle concentration' -> c_s_p_xav
    # 'Electrolyte concentration' -> c_e
    @variables Q(..) c_s_n_xav(..) c_s_p_xav(..) c_e(..)
    Dt = Differential(t)
    Drn = Differential(rn)
    Drp = Differential(rp)
    Dx = Differential(x)

    # 'X-averaged negative particle concentration' equation
    cache_6338548344403035350 = 8.813457647415216 * (1 / rn^2 * Drn(rn^2 * Drn(c_s_n_xav(t, rn))))

    # 'X-averaged positive particle concentration' equation
    cache_m4837956127753503843 = 22.598609352346717 * (1 / rp^2 * Drp(rp^2 * Drp(c_s_p_xav(t, rp))))

    # 'Electrolyte concentration' equation

    function concatenation(n, s, p)
    # A concatenation in the electrolyte domain
    IfElse.ifelse(
        x < 0.4444444444444445, n, IfElse.ifelse(
            x < 0.5555555555555556, s, p
        )
    )
    end

    cache_m3686175401074329161 = concatenation(-414.806227359111, -124.4418682077333, -414.806227359111)
    cache_8207698826108331396 = concatenation(-0.31475547656191716, -1.9155408290138962, -0.31475547656191716)
    cache_2712683893195935317 = concatenation(0.1806862603261852 * x, 0.08030500458941565, 
    0.1806862603261852 * (1.0 - x)
    )
    cache_m3812278407756951412 = concatenation(187.37446495382773, 0.0, -187.37446495382773)
    cache_m692082021805070208 = (cache_m3686175401074329161 * (Dx(((cache_8207698826108331396 * (exp(-0.65 * c_e(t, x)))) * Dx(c_e(t, x))) + cache_2712683893195935317))) + cache_m3812278407756951412


    eqs = [
    Dt(Q(t)) ~ 4.27249308415467,
    Dt(c_s_n_xav(t, rn)) ~ cache_6338548344403035350,
    Dt(c_s_p_xav(t, rp)) ~ cache_m4837956127753503843,
    Dt(c_e(t, x)) ~ cache_m692082021805070208,
    ]

    ics_bcs = [
    # initial conditions
    Q(0) ~ 0.0,
    c_s_n_xav(0, rn) ~ 0.8000000000000016,
    c_s_p_xav(0, rp) ~ 0.6000000000000001,
    c_e(0, x) ~ 1.0,
    # boundary conditions
    Drn(c_s_n_xav(t, 0.0)) ~ 0.0,
    Drn(c_s_n_xav(t, 1.0)) ~ -0.14182855923368468,
    Drp(c_s_p_xav(t, 0.0)) ~ 0.0,
    Drp(c_s_p_xav(t, 1.0)) ~ 0.03237700710041634,
    Dx(c_e(t, 0.0)) ~ 0.0,
    Dx(c_e(t, 1.0)) ~ 0.0,
    ]

    t_domain = IntervalDomain(0, 0.159)
    rn_domain = IntervalDomain(0.0, 1.0)
    rp_domain = IntervalDomain(0.0, 1.0)
    x_domain = IntervalDomain(0.0, 1.0)

    domains = [
    t in t_domain,
    rn in rn_domain,
    rp in rp_domain,
    x in x_domain,
    ]
    ind_vars = [t, rn, rp, x]
    dep_vars = [Q, c_s_n_xav, c_s_p_xav, c_e]

    SPMe_pde_system = PDESystem(eqs, ics_bcs, domains, ind_vars, dep_vars)
    end



    ## PINN Part


    begin
    num_dim = 50
    nonlin = Flux.gelu
    #in_dim = 3
    #out_dim = 3
    strategy = NeuralPDE.QuadratureTraining(;quadrature_alg=HCubatureJL(),abstol=1e-5, reltol=1, maxiters=1000, batch=0)
    #strategy_ = NeuralPDE.QuadratureTraining(;abstol=1e-6, reltol=1e-8, maxiters=2000)
    #strategy = NeuralPDE.QuadratureTraining(;quadrature_alg=HCubatureJL(), batch=0)
    #strategy = NeuralPDE.StochasticTraining(128)
    #strategy = NeuralPDE.QuadratureTraining()
    #strategy = NeuralPDE.QuadratureTraining(;quadrature_alg=HCubatureJL(),
                                                        #reltol=1e-3,abstol=1e-5,
                                                        #maxiters =2000, batch=0)

    #in_dims = [3, 3, 3]
    #in_dims = [1, 2, 2]
    in_dims = [2, 2]
    num_hid = 2
    chains_ = [FastChain(FastDense(in_dim,num_dim,nonlin),
                        [FastDense(num_dim,num_dim,nonlin) for i in 1:num_hid]...,
                        FastDense(num_dim,1)) for in_dim in in_dims]
    #adalosspoisson = NeuralPDE.LossGradientsAdaptiveLoss(20; α=0.9f0)
    adaloss = NeuralPDE.MiniMaxAdaptiveLoss(20; α_pde=0.3, α_bc=1e2, pde_weights_start=1e-3, bc_weights_start=1e3)
    data_dir = "/home/zobot/.julia/dev/NeuralPDEWatson/data"
    exp_name = "reduced_c_phi_adam_minimax"
    exp_folder = joinpath(data_dir, exp_name)
    wipe_logs = false
    discretization = NeuralPDE.PhysicsInformedNN(chains_,
                                                    strategy; adaptive_loss=adaloss, exp_name=exp_name, data_dir=data_dir, wipe=wipe_logs)
end
    sym_prob = NeuralPDE.symbolic_discretize(reduced_c_phi_pde_system,discretization)
    prob = NeuralPDE.discretize(reduced_c_phi_pde_system,discretization)
begin
                                                    
    end

    #@run sym_prob = NeuralPDE.symbolic_discretize(SPM_pde_system,discretization)
    #@run sym_prob = NeuralPDE.symbolic_discretize(SPM_pde_system,discretization)
    begin
    initθ = vcat(discretization.init_params...)
    #opt = Flux.Optimiser(ClipValue(1e-3), ExpDecay(1, 0.5, 25_000), ADAM(3e-4))
    opt = ADAM(3e-4)
    saveevery = 100
    loss = zeros(Float64, saveevery)

    losssavefile = joinpath(exp_folder, "loss.csv")

    if wipe_logs
        rm(losssavefile; force=true)
    end
    iteration_count_arr = [1]
    cb = function (p,l)
        iteration_count = iteration_count_arr[1]


        println("Current loss is: $l, iteration is: $(iteration_count)")
        loss[((iteration_count - 1) % saveevery) + 1] = l
        if iteration_count % saveevery == 0
            cursavefile = joinpath(exp_folder, string(iteration_count, base=10, pad=5) * ".csv")
            writedlm(cursavefile, p, ",")
            df = DataFrame(loss=loss)
            CSV.write(losssavefile, df, writeheader=false, append=true)
        end
        iteration_count_arr[1] += 1
        return false
    end
    end
    prob.f(initθ, [])
    res = GalacticOptim.solve(prob, ADAM(3e-4); cb = cb, maxiters=20_000)
end
#=
    pretrained_params_file = joinpath(exp_folder, "20000.csv")
    pretrained_pinn_params = Array(CSV.read(pretrained_params_file, DataFrame; header=false))[:,1]
    params = pretrained_pinn_params

    dts = sols_t
    drns = sols_r_n
    drps = sols_r_p

    #param_lengths = (length ∘ initial_params).(chains_)
    #indices_in_params = map(zip(param_lengths, cumsum(param_lengths))) do (param_length, cumsum_param)
            #cumsum_param - (param_length - 1) : cumsum_param
    #end

    #phi_i_params = [params[indices_in_params_i] for indices_in_params_i in indices_in_params]

    #Q_evals = [chains_[1]([dt], phi_i_params[1])[1] for dt in sols_t]

    #c_s_n_evals_pretrained_pinn = [chains_[2]([dt, drn], phi_i_params[2])[1] for dt in dts, drn in drns]
    c_s_p_evals_pretrained_pinn = [chains_[1]([dt, drp], params)[1] for dt in dts, drp in drps]

    anim = @animate for i in 1:length(dts)
        #p1 = plot(drns,sols_c_s_n[:, i];ylims=(0,1.3),ylabel="c_s_n",xlabel="r_n",legend=true, label="FDM")
        #p1 = plot(drns,c_s_n_evals_pretrained_pinn[i, :];ylims=(0,1.3),ylabel="c_s_n",xlabel="r_n",legend=true, label="FDM")
        #p1 = plot!(p1,drns,c_s_n_evals_pretrained_pinn[i,:];ylims=(0,1.3),ylabel="c_s_n",xlabel="r_n",legend=true, label="Fulltrained pinn", linestyle=:dashdot)
        p2 = plot(drps,sols_c_s_p[:, i];ylims=(0,1.3),ylabel="c_s_p",xlabel="r_p",legend=true, label="FDM")
        #p2 = plot(drps,c_s_p_evals_pretrained_pinn[i, :];ylims=(0,1.3),ylabel="c_s_p",xlabel="r_p",legend=true, label="FDM")
        p2 = plot!(p2,drps,c_s_p_evals_pretrained_pinn[i,:];ylims=(0,1.3),ylabel="c_s_p",xlabel="r_p",legend=true, label="Fulltrained pinn", linestyle=:dashdot)
        #plot(p1,p2)
    end
    gif(anim, joinpath(exp_folder, "SPM_fulltrain_adam2.gif"),fps=30)
end
=#
main_train()