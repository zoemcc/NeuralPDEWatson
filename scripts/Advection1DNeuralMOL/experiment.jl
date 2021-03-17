using DrWatson
import DrWatson.savename
@quickactivate "NeuralPDEWatson"
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

const C_adv = 2
const C_diff = 0

const x_max = 2
const t_max = 1



function advection_pinn()
    @parameters t, x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Dx^2



    # Equations, initial and boundary conditions ###############################

    eqs = [
                Dt(u(t, x)) + C_adv * Dx(u(t, x)) ~ 0 # C_diff * Dxx(u(t, x)) 
                ]

    
    
    bcs = [
                u(t, 0.0) ~ u(t, x_max),
                u(0.0, x) ~ cos(2π * (x / x_max))
                ]
    @show bcs

    # Space and time domains ###################################################

    domains = [
                t ∈ IntervalDomain(0, t_max),
                x ∈ IntervalDomain(0, x_max)
                ]




    strategy = NeuralPDE.QuadratureTraining(quadrature_alg=HCubatureJL(),
        reltol = 1e-5, abstol = 1e-4, maxiters = 300)

    @show strategy

    dim = length(domains)
    outputdim = length(eqs)
    circletransform(tx, θ) = [tx[1], cos(π*tx[2]), sin(π*tx[2])]


    node_hidden_dims = 25
    node_state_dims = 10

    node_state_to_params_dims = 25

    xdims = 1
    x_to_u_hid_dims = 32

    x_to_u = FastChain(FastDense(xdims, x_to_u_hid_dims, gelu), FastDense(x_to_u_hid_dims, x_to_u_hid_dims, gelu), FastDense(x_to_u_hid_dims, 1))
    x_to_u_params_dims = length(initial_params(x_to_u))

    tspan = Float32.((0, t_max))
    dudt = FastChain(FastDense(node_state_dims,node_hidden_dims,gelu),FastDense(node_hidden_dims,node_state_dims))
    node = NeuralODE(dudt,tspan,Tsit5())
    θ = initial_params(dudt)
    initial_node_state = randn(Float32, node_state_dims)

    node(initial_node_state)

    node_state_to_x_to_u_params = FastChain(FastDense(node_state_dims, node_state_to_params_dims, gelu), FastDense(node_state_to_params_dims, node_state_to_params_dims, gelu), FastDense(node_state_to_params_dims, x_to_u_params_dims))

    Flux.params(node)

    # this is the beginning of a single eval:

    tx = [0.5f0, 1.0f0]

    node_state_to_x_to_u_params_params = initial_params(node_state_to_x_to_u_params)
    fullp = cat(node.p, node_state_to_x_to_u_params_params, dims=1)
    node_p_length = length(node.p)
    nodep = node.p

    node(initial_node_state, zeros(Float32, node_p_length))

    nmol = NeuralMOL(initial_node_state, node, dudt, node_state_to_x_to_u_params, x_to_u, node_p_length)


    nmol(tx, fullp)

    nmol_fc = FastChain(nmol)

    nmol_fc(tx, fullp)

    initial_params(nmol)

    ip = initial_params(nmol_fc)
    #Zygote.gradient((tx, p)->nmol_fc(tx, p)[1], [1.0f0, 1.0f0], ip)
    nmol_fc(tx, ip)


    ts = (0:0.01:1)
    xs = (0:0.01:2)

    #us = [nmol_fc([t, x], ip)[1] for t in ts, x in xs]

    #surfaceplot = GLMakie.surface(ts, xs, us)

    discretization = PhysicsInformedNN(nmol_fc, strategy)


    pde_system = PDESystem(eqs, bcs, domains, [t,x], [u])
    prob = discretize(pde_system, discretization)
    #analytical_prob = discretize(pde_system, discretization_analytical)
    experiment_path = datadir("neural_mol", "first_experiment")

    iteration_count = [1]
    saveevery = 10
    loss = Float64[]
    cb = function (p, l)
        #iteration_count = discretization.iteration_count[1]
        println("Current loss is: $l, iteration is: $(iteration_count)")
        push!(loss, l)
        if iteration_count[1] % saveevery == 0
            cursavefile = joinpath(experiment_path, string(iteration_count[1], base=10, pad=5) * ".csv")
            writedlm(cursavefile, p, ",")
            losssavefile = joinpath(experiment_path, "loss.csv")
            writedlm(losssavefile, loss, ",")
        end
        iteration_count[1] += 1

        return false
    end


    maxiters=20000
    #maxiters=40
    res = GalacticOptim.solve(prob, RADAM(0.1);cb=cb, maxiters=maxiters)
    us_after_training = [nmol_fc([t, x], res.u)[1] for t in ts, x in xs]

    surfaceplot = GLMakie.surface(ts, xs, us_after_training)
    return res



end



struct NeuralMOL{INIT, NODE, NODE_FC, S_TO_P_FC, X_TO_U_FC, SIZE} <: DiffEqFlux.FastLayer
    initial_node_state::INIT
    node::NODE
    node_fc::NODE_FC
    s_to_p_fc::S_TO_P_FC
    x_to_u_fc::X_TO_U_FC
    num_node_fc_params::SIZE
end

function (f::NeuralMOL)(x, p)
    t_cur = x[1]
    x_arr = x[2:2]

    node = NeuralODE(f.node_fc,(0.0f0, t_cur),Tsit5())
    nodesol = node(f.initial_node_state, @view p[1:f.num_node_fc_params])
    nodeeval = nodesol[end]

    x_to_u_params = f.s_to_p_fc(nodeeval, @view p[f.num_node_fc_params + 1:end])
    f.x_to_u_fc(x_arr, x_to_u_params)
end

DiffEqFlux.initial_params(f::NeuralMOL) = cat(initial_params.((f.node_fc, f.s_to_p_fc))..., dims=1)
DiffEqFlux.paramlength(f::NeuralMOL) = length(initial_params(f))

