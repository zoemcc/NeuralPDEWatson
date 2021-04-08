include("experiment.jl")
using ImageMagick
using GeometryBasics
using AbstractPlotting

fnty(x) = fieldnames(typeof(x))
ty(x) = typeof(x)


function process_example_param()
    println("Loading parameters")
    experiment_title = "first_experiment"
    #experiment_params_str = savename(experiment_params)
    #experiment_params = parse_savename_to_experiment(ExperimentParams, experiment_params_str)
    video_extension = ".mp4"
    video_filename = datadir("neural_mol", experiment_title, "first_run" * video_extension)
    if isfile(video_filename)
        println("already videoed ")
        return nothing
    end
    # idempotent
    #@assert experiment_params_str == savename(experiment_params)
    experiment_dir = datadir("neural_mol", experiment_title)

    parameter_filenames = sort(filter(x->occursin(r"[0-9][0-9][0-9][0-9][0-9].csv", x), readdir(experiment_dir)))

    filename_to_parameter_vector(parameter_filename) = Array(CSV.read(joinpath(experiment_dir, parameter_filename), DataFrame; header=false))[:, 1]

    parameter_vectors = filename_to_parameter_vector.(parameter_filenames)

    dim = 1
    outputdim = 1


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
    chain = nmol_fc

    isnanparameters = map(vec->any(isnan.(vec)), parameter_vectors)
    if any(isnanparameters)
        @show isnanparameters
        firstnanindex = findfirst(isnanparameters)
        println("$(firstnanindex) is the first nan index, filename: $(parameter_filenames[firstnanindex])")
        println("Nans detected, saving only up to the first nan")
        parameter_vectors = parameter_vectors[1:firstnanindex-1]
    end
    function analytical_solution(v::AbstractArray) 
        t, x = v[1], v[2]
        cos(2π * ( x - C_adv * t ) / x_max)
    end
    analytical_solution(v::AbstractArray, θ::AbstractArray) = analytical_solution(v)
    analytical_solution_fastchain = FastChain( analytical_solution )

    xs = (0.0:0.01:x_max)
    ts = (0.0:0.01:t_max)
    println("Evaluating function for each intermediate parameter")
    zs = [zeros(Float64, 1, 1) for θ in parameter_vectors]
    diffs = [zeros(Float64, 1, 1) for θ in parameter_vectors]
    analyticals = [analytical_solution([t, x]) for t in ts, x in xs]
    for i in 1:length(parameter_vectors)
        zs[i] = [chain([t, x], parameter_vectors[i])[1] for t in ts, x in xs]
        diffs[i] = abs.(zs[i] .- analyticals)
    end


    function integrate_difference(θ)
        integrand(tx, p) = (chain(tx, θ)[1] - analytical_solution(tx)) ^ 2
        quadprob = QuadratureProblem(integrand, SA[0.0, 0.0], SA[1.0, 2.0])
        solve(quadprob, HCubatureJL(), reltol=1e-3, abstol=1e-4).u
    end

    #last_parameter = parameter_vectors[end]
    #integrate_difference(last_parameter)
    #vector_differences = integrate_difference.(parameter_vectors)
    println("Evaluating quadrature of difference of learned solution and the analytical solution")
    quadrature_difference_throughout_training = zeros(Float64, length(parameter_vectors))
    Threads.@threads for i in 1:length(parameter_vectors)
        quadrature_difference_throughout_training[i] = integrate_difference(parameter_vectors[i])
    end
    #Plots.plot(vector_differences)
    #Plots.plot(quadrature_difference_throughout_training)

    quadrature_differences_file = datadir("neural_mol", experiment_title, "quadratures" * ".csv")
    writedlm(quadrature_differences_file, quadrature_difference_throughout_training, ",")




    #f(t, x) = chain([t, x], params)[1]
    #zs = [[chain([t, x], θ)[1] for t in ts, x in xs] for θ in parameter_vectors]
    #plot(ts, xs, f, st=:surface)
    println("Generating video of training") 
    fig = MakieLayout.Figure(resolution=(1920,1080), title="Learning ϕ for 1D Advection PDE Approximate Solution")


    #lscene = LScene(fig[2,1], scenekw = (camera = cam3d!, raw=false); height=650, width=450, tellwidth=false, tellheight=false, alignmode=Outside())
    lscene = LScene(fig[3,1], scenekw = (camera = cam3d!, raw=false))
    surfscene = GLMakie.surface!(lscene, ts, xs, analyticals; colorrange=(-1, 1))

    surfaxis = lscene.scene.plots[1]
    surfaxis.attributes.names.axisnames[] = ("t", "x", "ϕ")
    surfaxis.input_args[1][] = GeometryBasics.HyperRectangle{3,Float32}(Float32[-0.05, -0.05, -1.1], Float32[1.1, 2.1, 2.2])
    surfobs = lscene.scene.plots[2].input_args[3]

    #colsize!(fig.layout, 1, Fixed(500))

    #lscenediff = LScene(fig[2,2], scenekw = (camera = cam3d!, raw=false); height=650, width=450, tellwidth=false, tellheight=false, alignmode=Outside())
    lscenediff = LScene(fig[3,2], scenekw = (camera = cam3d!, raw=false))
    surfdiffscene = GLMakie.surface!(lscenediff, ts, xs, abs.(1.0f0 .- analyticals); colorrange=(0, 2))
    surfaxisdiff = lscenediff.scene.plots[1]
    surfaxisdiff.attributes.names.axisnames[] = ("t", "x", "|Δϕ|")
    surfaxisdiff.input_args[1][] = GeometryBasics.HyperRectangle{3,Float32}(Float32[-0.05, -0.05, -0.1], Float32[1.1, 2.1, 2.2])
    surfobsdiff = lscenediff.scene.plots[2].input_args[3]

    lsceneana = LScene(fig[3,3], scenekw = (camera = cam3d!, raw=false))
    surfsceneana = GLMakie.surface!(lsceneana, ts, xs, analyticals; colorrange=(-1, 1))

    surfaxisana = lsceneana.scene.plots[1]
    surfaxisana.attributes.names.axisnames[] = ("t", "x", "analytical")
    surfaxisana.input_args[1][] = GeometryBasics.HyperRectangle{3,Float32}(Float32[-0.05, -0.05, -1.1], Float32[1.1, 2.1, 2.2])
    #surfobs = lscene.scene.plots[2].input_args[3]

    #colsize!(fig.layout, 2, Fixed(500))
    #rowsize!(fig.layout, 2, Fixed(700))

    hm_bottom_sublayout = GridLayout()
    fig[4,1:3] = hm_bottom_sublayout
    #hm_bottom_sublayout.height = 300

    contourlevels = (-1f0:0.2f0:1f0)
    length(contourlevels)

    axiscontour = hm_bottom_sublayout[1, 1] = Axis(fig, title="ϕ vs t-x")
    contourscene = GLMakie.contourf!(axiscontour, ts, xs, zs[1]; colorrange=(-1, 1), levels=contourlevels, extendhigh=:auto, extendlow=:auto)
    axiscontour.xlabel="t"
    axiscontour.ylabel="x"
    contourplot = axiscontour.scene.plots[1]
    contourobs = contourplot.input_args[3]
    colorbar = hm_bottom_sublayout[1, 2] = Colorbar(fig, contourscene; width=10)


    axisdiffcontour = hm_bottom_sublayout[1, 3] = Axis(fig, title="|ϕ - analytical| vs t-x")
    contourdiffscene = GLMakie.contourf!(axisdiffcontour, ts, xs, diffs[1]; colorrange=(0, 2), levels=(contourlevels .+ 1f0), extendhigh=:auto, extendlow=:auto)
    axisdiffcontour.xlabel="t"
    axisdiffcontour.ylabel="x"
    contourdiffplot = axisdiffcontour.scene.plots[1]
    contourdiffobs = contourdiffplot.input_args[3]
    colorbar_diff = hm_bottom_sublayout[1, 4] = Colorbar(fig, contourdiffscene; width=10)


    axisanacontour = hm_bottom_sublayout[1, 5] = Axis(fig, title="analytical vs t-x")
    contouranascene = GLMakie.contourf!(axisanacontour, ts, xs, analyticals; colorrange=(-1, 1), levels=contourlevels, extendhigh=:auto, extendlow=:auto)
    axisanacontour.xlabel="t"
    axisanacontour.ylabel="x"
    contouranaplot = axisanacontour.scene.plots[1]
    colorbar_ana = hm_bottom_sublayout[1, 6] = Colorbar(fig, contouranascene; width=10)


    titleiter = fig[1,:] = Label(fig, "Learning ϕ for 1D Advection PDE Approximate Solution Iteration: 0", textsize=30)
    titleparams = fig[2,:] = Label(fig, "Params: nmol first", textsize=10)
    rowsize!(fig.layout, 4, Fixed(200))
    #rowsize!(fig.layout, 1, Fixed(100))

    display(fig)

    #lscenediff.scene.camera.eyeposition



    focal_point = Vec3f0(0.5f0, 1.0f0, 0.0f0)
    vertical_offset = Vec3f0(0, 0, 1)
    record(fig, video_filename; framerate=30, compression=30) do io
        for i in 1:length(parameter_vectors)
            cam_position = Vec3f0(focal_point .+ 1.0 .* Vec3f0(3 * cos(i/30), 4 * sin(i/30), 2.0 + 3.0 * sin(i/50)))
            
            imod = ((1 * i - 1) % length(parameter_vectors)) + 1
            surfobs.val[:] .= @view zs[imod][:]
            contourobs.val[:] .= @view zs[imod][:]
            surfobsdiff.val[:] .= @view diffs[imod][:]
            contourdiffobs.val[:] .= @view diffs[imod][:]
            Observables.notify!(surfobs)
            Observables.notify!(contourobs)
            Observables.notify!(surfobsdiff)
            Observables.notify!(contourdiffobs)
            AbstractPlotting.update_cam!(lscene.scene, cam_position, focal_point)
            AbstractPlotting.update_cam!(lscenediff.scene, cam_position .+ vertical_offset, focal_point .+ vertical_offset)
            AbstractPlotting.update_cam!(lsceneana.scene, cam_position, focal_point)
            titleiter.elements[:text].text[] = "Learning ϕ for 1D Advection PDE Approximate Solution Iteration: $(i * 10)"
            sleep(0.01)
            recordframe!(io)
            sleep(0.01)
        end
    end
    

end


function process_all_finished_experiments(experiment_title::AbstractString)
    experiment_list = filter(s->occursin(r"ChainDims=", s), readdir(datadir("hyperopt", experiment_title)))
    
    finished_experiment_list = filter(s->is_experiment_finished(s, experiment_title), experiment_list)
    println("number of finished experiments to video: $(length(finished_experiment_list))")
    for experiment_str in finished_experiment_list

        #experiment_params = parse_savename_to_experiment(ExperimentParams, experiment_str)
        println("processing: $(experiment_str)")
        process_example_param(experiment_str)

    end


end

#const experiment_title = "AdvectionOnly1DBigSearch"

#process_all_finished_experiments(experiment_title)
