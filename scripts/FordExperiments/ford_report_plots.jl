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
end

begin
pybamm_sols = myunpickle("/home/zobot/.julia/dev/DFN.jl/pybamm/hardcoded_models/MTK_format/pybamm_solutions/SPM.pickle")

time_slice = 1:94
sols_t = pybamm_sols["Time"][time_slice]
sols_r_n = pybamm_sols["r_n"][:,1]
sols_r_p = pybamm_sols["r_p"][:,1]
sols_c_s_n = pybamm_sols["X-averaged negative particle concentration"][:, time_slice]
sols_c_s_p = pybamm_sols["X-averaged positive particle concentration"][:, time_slice]

in_dims = [1, 2, 2]
input_dims = [[1], [1,2], [1,3]]
num_hid = 2
num_dim = 50
nonlin = Flux.gelu
chains_ = [FastChain(FastDense(in_dim,num_dim,nonlin),
                    [FastDense(num_dim,num_dim,nonlin) for i in 1:num_hid]...,
                    FastDense(num_dim,1)) for in_dim in in_dims]

param_lengths = (length ∘ initial_params).(chains_)
indices_in_params = map(zip(param_lengths, cumsum(param_lengths))) do (param_length, cumsum_param)
        cumsum_param - (param_length - 1) : cumsum_param
end

data_dir = "/home/zobot/.julia/dev/NeuralPDEWatson/data"
exp_folder_nonadaptive = joinpath(data_dir, "minimax_consistent_nonadaptive")
exp_folder_minimax = joinpath(data_dir, "minimax_consistent")
exp_folder_loss_gradients = joinpath(data_dir, "minimax_consistent_loss_gradients")

function eval_params(exp_folder)
    params_file = joinpath(exp_folder, "50000.csv")
    params = Array(CSV.read(params_file, DataFrame; header=false))[:,1]
    params_arr = [params[indices_in_params_i] for indices_in_params_i in indices_in_params]
    Q_evals = [chains_[1]([dt], params_arr[1])[1] for dt in sols_t]
    c_s_n_evals = [chains_[2]([dt, drn], params_arr[2])[1] for dt in sols_t, drn in sols_r_n]
    c_s_p_evals = [chains_[3]([dt, drp], params_arr[3])[1] for dt in sols_t, drp in sols_r_p]
    (Q_evals=Q_evals, c_s_n_evals=c_s_n_evals, c_s_p_evals=c_s_p_evals)
end


nonadaptive_evals, minimax_evals, loss_gradients_evals = map(eval_params, [exp_folder_nonadaptive, exp_folder_minimax, exp_folder_loss_gradients])


output_location = "/home/zobot/.julia/dev/WorkNotes/PaperDrafts/FordEarlyJun2021/"
function subplot_differences_rn(evals, name)
    p1 = contourf(sols_r_n, sols_t, sols_c_s_n', title="PyBaMM", xaxis="r_n", yaxis="t")
    p2 = contourf(sols_r_n, sols_t, evals.c_s_n_evals, title=name, xaxis="r_n", yaxis="t")
    error = abs.(evals.c_s_n_evals .- sols_c_s_n')
    p3 = contourf(sols_r_n, sols_t, error, title="error", xaxis="r_n", yaxis="t")
    #l = @layout [a b; c]
    finalplot = plot(p1, p2, p3)
    output_name = name * "_r_n_spm.pdf"
    @show output_name
    @show norm(error) / norm(sols_c_s_n) * 100
    savefig(finalplot, joinpath(output_location, output_name))
    finalplot
end
function subplot_differences_rp(evals, name)
    p1 = contourf(sols_r_p, sols_t, sols_c_s_p', title="PyBaMM", xaxis="r_p", yaxis="t")
    p2 = contourf(sols_r_p, sols_t, evals.c_s_p_evals, title=name, xaxis="r_p", yaxis="t")
    error = abs.(evals.c_s_p_evals .- sols_c_s_p')
    p3 = contourf(sols_r_p, sols_t, error, title="error", xaxis="r_p", yaxis="t")
    #l = @layout [a b; c]
    finalplot = plot(p1, p2, p3)
    output_name = name * "_r_p_spm.pdf"
    @show output_name
    @show norm(error) / norm(sols_c_s_p) * 100
    savefig(finalplot, joinpath(output_location, output_name))
    finalplot
end

output_plots_rn = map(subplot_differences_rn, [nonadaptive_evals, minimax_evals, loss_gradients_evals], ["nonadaptive", "minimax", "loss_gradients"])
output_plots_rp = map(subplot_differences_rp, [nonadaptive_evals, minimax_evals, loss_gradients_evals], ["nonadaptive", "minimax", "loss_gradients"])

end

begin
pybamm_sols = myunpickle("/home/zobot/.julia/dev/DFN.jl/pybamm/hardcoded_models/MTK_format/pybamm_solutions/reduced_c_phi.pickle")

t = pybamm_sols["Time"]
x = pybamm_sols["x"][:,1]
c_e = pybamm_sols["Electrolyte concentration"]
phi_e = pybamm_sols["Electrolyte potential"]

in_dims = [2, 2]
input_dims = [[1,2], [1,2]]
num_hid = 2
num_dim = 50
nonlin = Flux.gelu
chains_ = [FastChain(FastDense(in_dim,num_dim,nonlin),
                    [FastDense(num_dim,num_dim,nonlin) for i in 1:num_hid]...,
                    FastDense(num_dim,1)) for in_dim in in_dims]

param_lengths = (length ∘ initial_params).(chains_)
indices_in_params = map(zip(param_lengths, cumsum(param_lengths))) do (param_length, cumsum_param)
        cumsum_param - (param_length - 1) : cumsum_param
end

data_dir = "/home/zobot/.julia/dev/NeuralPDEWatson/data"
params_file_minimax = joinpath(data_dir, "reduced_c_phi_adam_minimax", "13400.csv")
params_file_lossgradients = joinpath(data_dir, "reduced_c_phi_adam_lossgradients", "20000.csv")

function eval_params(params_file)
    params_read = Array(CSV.read(params_file, DataFrame; header=false))[:,1]
    params_arr = [params_read[indices_in_params_i] for indices_in_params_i in indices_in_params]
    c_e = [chains_[1]([dt, dx], params_arr[1])[1] for dt in t, dx in x]
    phi_e = [chains_[2]([dt, dx], params_arr[2])[1] for dt in t, dx in x]
    (c_e=c_e, phi_e=phi_e)
end


minimax_evals, loss_gradients_evals = map(eval_params, [params_file_minimax, params_file_lossgradients])


output_location = "/home/zobot/.julia/dev/WorkNotes/PaperDrafts/FordEarlyJun2021/"
name = "minimax"
evals = minimax_evals
function subplot_differences_c_e(evals, name)
    p1 = contourf(t, x, c_e, title="PyBaMM", xaxis="t", yaxis="x")
    p2 = contourf(t, x, evals.c_e', title=name, xaxis="t", yaxis="x")
    error = abs.(evals.c_e' .- c_e)
    p3 = contourf(t, x, error, title="error", xaxis="t", yaxis="x")
    #l = @layout [a b; c]
    finalplot = plot(p1, p2, p3)
    output_name = name * "_c_e_reduced_c_phi.pdf"
    @show output_name
    @show norm(error) / norm(c_e) * 100
    savefig(finalplot, joinpath(output_location, output_name))
    finalplot
end
function subplot_differences_phi_e(evals, name)
    p1 = contourf(t, x, phi_e, title="PyBaMM", xaxis="t", yaxis="x")
    p2 = contourf(t, x, evals.phi_e', title=name, xaxis="t", yaxis="x")
    error = abs.(evals.phi_e' .- phi_e)
    p3 = contourf(t, x, error, title="error", xaxis="t", yaxis="x")
    #l = @layout [a b; c]
    finalplot = plot(p1, p2, p3)
    output_name = name * "_phi_e_reduced_c_phi.pdf"
    @show output_name
    @show norm(error) / norm(phi_e) * 100
    savefig(finalplot, joinpath(output_location, ))
    finalplot
end

output_plots_c_e = map(subplot_differences_c_e, [minimax_evals, loss_gradients_evals], ["minimax", "loss_gradients"])
output_plots_phi_e = map(subplot_differences_phi_e, [minimax_evals, loss_gradients_evals], ["minimax", "loss_gradients"])

end