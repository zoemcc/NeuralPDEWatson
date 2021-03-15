using DrWatson
import DrWatson.savename
@quickactivate "NeuralPDEWatson"
DrWatson.greet()

using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
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

const C_adv = 2.0
const C_diff = 0.0

const x_max = 2.0
const t_max = 1.0


@with_kw struct ExperimentParams
    ChainDims::Int64
    ChainNonLin::Symbol
    UseCircleTransform::Bool
    NumHidLayers::Int64
    L2RegularizationWeight::Float64
    FinalPDELossWeight::Float64
    PDELossSchedule::Symbol
    LearningRate::Float64
    OptimizationMethod::Symbol
    Strategy::Symbol
end



function ensuredirexists(directory)
    if !isdir(directory) 
        mkdir(directory)
    end
    directory
end

function main_hyperopt_run(n::Integer; do_generate_hyperopt_params=false, dry_run=true)
    experiment_title = "AdvectionOnly1DBigSearch"
    allowed_parameter_variations = Dict(
        :ChainDims => (64, 32),
        :ChainNonLin => (:relu, :gelu, :sigmoid),
        :UseCircleTransform => (true, false),
        :NumHidLayers => (2, 3, 4),
        :L2RegularizationWeight => (1e-4, 1e-5, 1e-6),
        :FinalPDELossWeight => (1e-2, 1e-1, 1e0, 1e1),
        :PDELossSchedule => (:long, :short, :none),
        :LearningRate => (1e-2, 3e-2, 1e-1),
        :OptimizationMethod => (:ADAM, :RADAM, :GD),
        :Strategy => (:Grid1, :Stochastic128, :Quadrature60, :Quadrature200, :QuasiRandom100),
    )

    # generate hyperopt candidates if none exist, save them to list in a folder
    ensuredirexists(datadir("hyperopt", experiment_title)) 
    experiment_list_filename = datadir("hyperopt", experiment_title, "experiment_list.csv")

    if do_generate_hyperopt_params || !isfile(experiment_list_filename)
        experiment_hyperopts = generate_hyperopt_params(n, allowed_parameter_variations)
        experiment_strings = savename.(experiment_hyperopts)
        writedlm(experiment_list_filename, experiment_strings, "\n")
    end

    experiment_strings = Array(CSV.read(experiment_list_filename, DataFrame; header=false))[:, 1]
    Threads.@threads for experiment_string in experiment_strings
        experiment_params = parse_savename_to_experiment(ExperimentParams, experiment_string)

        if !is_experiment_finished(experiment_string, experiment_title)
            @show "unfinished", experiment_params
            experiment_dir = ensuredirexists(datadir("hyperopt", experiment_title, experiment_string))

            train_advection_model(experiment_params, experiment_dir; dry_run=dry_run)

            # finish marker
            if !dry_run
                touch(datadir("hyperopt", experiment_title, experiment_string, "finished.txt"))
            end

        end

    end

end

function is_experiment_finished(experiment_string::AbstractString, experiment_title::AbstractString)::Bool
    isfile(datadir("hyperopt", experiment_title, experiment_string, "finished.txt"))
end


function generate_hyperopt_params(n::Integer, allowed_parameter_variations::Dict)
    ho = Hyperoptimizer(n; allowed_parameter_variations...)
    experiments = ExperimentParams[]
    for example in ho
        experiment_params_nt = split(example, :i)[2]
        experiment_params = ExperimentParams(; experiment_params_nt...)
        push!(experiments, experiment_params)
    end
    experiments
end





savename(experimentparams::ExperimentParams) = DrWatson.savename(struct2dict(experimentparams); digits=20)


#=
example_params = ExperimentParams(ChainDims=64, 
    ChainNonLin=:gelu,
    UseCircleTransform=true,
    NumHidLayers=4,
    L2RegularizationWeight=1e-5,
    FinalPDELossWeight=1e-1,
    PDELossSchedule=:short,
    LearningRate=1e-2,
    OptimizationMethod=:RADAM,
    Strategy=:Quadrature200,
)

example_savedname = savename(example_params)
=#

function parse_savename_to_experiment(paramstype::DataType, savedname::AbstractString; has_title=false)
    experiment_param_strings = split(savedname, "_")
    if has_title
        title = experiment_param_strings[1]
        experiment_param_strings = experiment_param_strings[2:end]
    else
        title = ""
    end

    param_fields = fieldnames(paramstype)
    param_types = fieldtypes(paramstype)
    param_fields_types = Dict(zip(param_fields, param_types))

    function string_to_param_pair(experiment_param_string::AbstractString)
        experiment_param_name_string, experiment_param_value_string = split(experiment_param_string, "=")
        experiment_param_name_symbol = Symbol(experiment_param_name_string)
        experiment_param_type = param_fields_types[experiment_param_name_symbol]
        experiment_param_value =  experiment_param_type != Symbol ? parse(experiment_param_type, experiment_param_value_string) : Symbol(experiment_param_value_string)
        experiment_param = experiment_param_name_symbol => experiment_param_value
    end

    param_nt = namedtuple(Dict(map(string_to_param_pair, experiment_param_strings)))
    paramstype(;param_nt...)
end


function chain_model(experiment_params::ExperimentParams)
    dim = 2
    outputdim = 1
    circletransform(tx, θ) = [tx[1], cos(π*tx[2]), sin(π*tx[2])]

    # make neural net

    chain_array = []
    if experiment_params.UseCircleTransform
        push!(chain_array, circletransform)
        inputdim = dim + 1
    else
        inputdim = dim
    end

    non_lin_functions = Dict(
        :gelu => Flux.gelu,
        :relu => Flux.relu,
        :sigmoid => Flux.σ,
    )

    push!(chain_array, FastDense(inputdim, experiment_params.ChainDims, non_lin_functions[experiment_params.ChainNonLin]))

    for i in 1:experiment_params.NumHidLayers - 1
        push!(chain_array, FastDense(experiment_params.ChainDims, experiment_params.ChainDims, non_lin_functions[experiment_params.ChainNonLin]))
    end

    push!(chain_array, FastDense(experiment_params.ChainDims, outputdim))

    chain = FastChain(chain_array...)


end

function analytical_solution(v::AbstractArray) 
    t, x = v[1], v[2]
    cos(2π * ( x - C_adv * t ) / x_max)
end
analytical_solution(v::AbstractArray, θ::AbstractArray) = analytical_solution(v)
analytical_solution_fastchain = FastChain( analytical_solution )


function train_advection_model(experiment_params::ExperimentParams, experiment_path::AbstractString; dry_run=true)

    @parameters t, x
    @variables u(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Dx^2



    # Equations, initial and boundary conditions ###############################

    eqs = [
                Dt(u(t, x)) + C_adv * Dx(u(t, x)) ~ 0 # C_diff * Dxx(u(t, x)) 
                ]

    
    
    if !experiment_params.UseCircleTransform
        bcs = [
                    u(t, 0.0) ~ u(t, x_max),
                    u(0.0, x) ~ cos(2π * (x / x_max))
                    ]
    else
        bcs = [
                    u(0.0, x) ~ cos(2π * (x / x_max))
                    ]

    end
    @show bcs

    # Space and time domains ###################################################

    domains = [
                t ∈ IntervalDomain(0.0, t_max),
                x ∈ IntervalDomain(0.0, x_max)
                ]




    StrategyDict = Dict(
        :Grid1 => NeuralPDE.GridTraining(0.1),
        :Stochastic128 => NeuralPDE.StochasticTraining(128),
        :Quadrature60 => NeuralPDE.QuadratureTraining(quadrature_alg=HCubatureJL(),
            reltol = 1e-5, abstol = 1e-4,
            maxiters = 60),
        :Quadrature200 => NeuralPDE.QuadratureTraining(quadrature_alg=HCubatureJL(),
            reltol = 1e-7, abstol = 1e-6,
            maxiters = 200),
        :QuasiRandom100 => NeuralPDE.QuasiRandomTraining(100; #points
                                                sampling_alg = UniformSample(),
                                                minibatch = 100),
    )

    strategy = StrategyDict[experiment_params.Strategy]
    @show strategy

    dim = length(domains)
    outputdim = length(eqs)
    circletransform(tx, θ) = [tx[1], cos(π*tx[2]), sin(π*tx[2])]

    # make neural net

    chain_array = []
    if experiment_params.UseCircleTransform
        push!(chain_array, circletransform)
        inputdim = dim + 1
    else
        inputdim = dim
    end

    non_lin_functions = Dict(
        :gelu => Flux.gelu,
        :relu => Flux.relu,
        :sigmoid => Flux.σ,
    )

    push!(chain_array, FastDense(inputdim, experiment_params.ChainDims, non_lin_functions[experiment_params.ChainNonLin]))
    for i in 1:experiment_params.NumHidLayers - 1
        push!(chain_array, FastDense(experiment_params.ChainDims, experiment_params.ChainDims, non_lin_functions[experiment_params.ChainNonLin]))
    end

    push!(chain_array, FastDense(experiment_params.ChainDims, outputdim))

    chain = FastChain(chain_array...)

    #@show length(chain_array)

    #loadpath = "/home/zobot/.julia/dev/NeuralPDEExamples/data/advection_diffusion_experiments/mar2_day_simple_circle_gelu_schedule/"
    #params = Array(CSV.read(loadpath * "00100.csv", DataFrame; header=false))[:, 1]
    #discretization = PhysicsInformedNN(chain1, strategy; init_params=params)
    #century_pde_loss_schedule = [0.0; 0.0; [10.0^-i for i in 5.0:-0.5:1.0]]

    century_pde_loss_schedules_dict = Dict(
        :none => [1.0],
        :short => [0.0, 0.1, 1.0],
        :long => [0.0, 0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    )
    century_pde_loss_schedule = (century_pde_loss_schedules_dict[experiment_params.PDELossSchedule] .* experiment_params.FinalPDELossWeight)
    #century_pde_loss_schedule = StaticArray(...)

    @show century_pde_loss_schedule
    @show experiment_params.L2RegularizationWeight


    discretization = PhysicsInformedNN(chain, strategy, iteration_count=[1]; century_pde_loss_schedule=century_pde_loss_schedule, l2_regularization_weight=experiment_params.L2RegularizationWeight, experiment_dir=experiment_path)
    #discretization = PhysicsInformedNN(chain, strategy; iter_count=[1], century_pde_loss_schedule=century_pde_loss_schedule, l2_regularization_weight=experiment_params.L2RegularizationWeight, experiment_dir="")
    #discretization_analytical = PhysicsInformedNN(analytical_solution_fastchain, strategy; iter_count=[1], century_pde_loss_schedule=century_pde_loss_schedule, l2_regularization_weight=experiment_params.L2RegularizationWeight, experiment_dir="")
    #discretization = PhysicsInformedNN(analytical_solution_fastchain, strategy; iter_count=[1], century_pde_loss_schedule=century_pde_loss_schedule, l2_regularization_weight=experiment_params.L2RegularizationWeight, experiment_dir="")


    pde_system = PDESystem(eqs, bcs, domains, [t,x], [u])
    prob = discretize(pde_system, discretization)
    #analytical_prob = discretize(pde_system, discretization_analytical)

    saveevery = 10
    loss = Float64[]
    cb = function (p, l)
        iteration_count = discretization.iteration_count[1]
        println("Current loss is: $l, iteration is: $(iteration_count)")
        push!(loss, l)
        if iteration_count[1] % saveevery == 0
            cursavefile = joinpath(experiment_path, string(iteration_count, base=10, pad=5) * ".csv")
            writedlm(cursavefile, p, ",")
            losssavefile = joinpath(experiment_path, "loss.csv")
            writedlm(losssavefile, loss, ",")
        end
        discretization.iteration_count[1] += 1

        return false
    end


    if !dry_run
        optimization_method_dict = Dict(
            :RADAM => RADAM(experiment_params.LearningRate),
            :ADAM => ADAM(experiment_params.LearningRate),
            :GD => Momentum(experiment_params.LearningRate),
        )
        println("Beginning to optimize problem:")
        optimization_method = optimization_method_dict[experiment_params.OptimizationMethod]
        #maxiters=5000
        maxiters=40
        res = GalacticOptim.solve(prob, optimization_method;cb=cb, maxiters=maxiters)
        return res
    else
        println("Skipping optimization")
        return nothing
    end


end




