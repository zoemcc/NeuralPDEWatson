include("advection_only.jl")
using ImageMagick
using GeometryBasics
using AbstractPlotting

fnty(x) = fieldnames(typeof(x))
ty(x) = typeof(x)

function grab_example_params()
    experiment_params_str = "ChainDims=32_ChainNonLin=gelu_FinalPDELossWeight=0.01_L2RegularizationWeight=1e-6_LearningRate=0.1_NumHidLayers=4_OptimizationMethod=GD_PDELossSchedule=short_Strategy=Quadrature60_UseCircleTransform=true"
    experiment_params = parse_savename_to_experiment(ExperimentParams, experiment_params_str)
    experiment_params_str, experiment_params
end

#process_example_param(experiment_params_str::AbstractString) = process_example_param(parse_savename_to_experiment(ExperimentParams, experiment_params_str))

example_params_str, example_params = grab_example_params()
#process_example_param(example_params_str)
const experiment_params_str = example_params_str

function process_example_param(experiment_params_str::AbstractString)
    println("Loading parameters")
    experiment_title = "AdvectionOnly1DBigSearch"
    #experiment_params_str = savename(experiment_params)
    experiment_params = parse_savename_to_experiment(ExperimentParams, experiment_params_str)
    video_extension = ".mp4"
    video_filename = datadir("hyperopt", experiment_title, "videos", experiment_params_str * video_extension)
    if isfile(video_filename)
        println("already videoed $(experiment_params_str)")
        return nothing
    end
    # idempotent
    #@assert experiment_params_str == savename(experiment_params)
    experiment_dir = datadir("hyperopt", experiment_title, experiment_params_str)

    parameter_filenames = sort(filter(x->occursin(r"[0-9][0-9][0-9][0-9][0-9].csv", x), readdir(experiment_dir)))

    filename_to_parameter_vector(parameter_filename) = Array(CSV.read(joinpath(experiment_dir, parameter_filename), DataFrame; header=false))[:, 1]

    parameter_vectors = filename_to_parameter_vector.(parameter_filenames)

    chain = chain_model(experiment_params)

    isnanparameters = map(vec->any(isnan.(vec)), parameter_vectors)
    if any(isnanparameters)
        @show isnanparameters
        firstnanindex = findfirst(isnanparameters)
        println("$(firstnanindex) is the first nan index, filename: $(parameter_filenames[firstnanindex])")
        println("Nans detected, saving only up to the first nan")
        parameter_vectors = parameter_vectors[1:firstnanindex-1]
    end

    xs = (0.0:0.01:x_max)
    ts = (0.0:0.01:t_max)
    println("Evaluating function for each intermediate parameter")
    zs = [zeros(Float64, 1, 1) for θ in parameter_vectors]
    diffs = [zeros(Float64, 1, 1) for θ in parameter_vectors]
    analyticals = [analytical_solution([t, x]) for t in ts, x in xs]
    Threads.@threads for i in 1:length(parameter_vectors)
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

    quadrature_differences_file = datadir("hyperopt", experiment_title, "quadratures", experiment_params_str * ".csv")
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
    titleparams = fig[2,:] = Label(fig, "Params: $(experiment_params_str)", textsize=10)
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
    
    #=
    scene = Scene()
    display(scene)
    surfscene = GLMakie.surface!(scene, ts, xs, zs[1])
    surfaxis = scene.plots[1]
    surfaxis.attributes.names.axisnames[] = ("t", "x", "u")
    surfaxis.input_args[1][] = GeometryBasics.HyperRectangle{3,Float32}(Float32[-0.05, -0.05, -1.1], Float32[1.1, 2.1, 2.2])
    surfobs = scene.plots[2].input_args[3]
    cam3d!(scene)
    #record_events(surfscene, datadir("hyperopt", experiment_title, "viewing_events")) do io
    focal_point = [0.5f0, 1.0f0, 0.0f0]
    ensuredirexists(datadir("hyperopt", experiment_title, "videos"))
    #mp = datadir("hyperopt", experiment_title, "videos", experiment_params_str * ".mp4")
    record(scene, video_filename; framerate=24, compression=20) do io
        for i in 1:499
            update_cam!(scene, cameracontrols(scene), Float32.(focal_point .+ [3 * cos(i/30), 4 * sin(i/30), 2.0 + 3.0 * sin(i/50)]), focal_point)
            imod = ((1 * i) % (length(parameter_vectors) - 1)) + 1
            surfobs.val[:] .= @view zs[imod][:]
            Observables.notify!(surfobs)
            sleep(0.01)
            recordframe!(io)
            sleep(0.01)
        end
    end
    =#


end

#=
begin 
    fig = MakieLayout.Figure(resolution=(1280,1280), title="Iteration 0")


    lscene = LScene(fig[2,1], scenekw = (camera = cam3d!, raw=false); height=650, width=450, tellwidth=false, tellheight=false, alignmode=Outside())
    surfscene = GLMakie.surface!(lscene, ts, xs, zs[1])

    surfaxis = lscene.scene.plots[1]
    surfaxis.attributes.names.axisnames[] = ("t", "x", "u")
    surfaxis.input_args[1][] = GeometryBasics.HyperRectangle{3,Float32}(Float32[-0.05, -0.05, -1.1], Float32[1.1, 2.1, 2.2])
    surfobs = lscene.scene.plots[2].input_args[3]
    camscene = cam3d!(lscene.scene)

    colsize!(fig.layout, 1, Fixed(500))

    lscenediff = LScene(fig[2,2], scenekw = (camera = cam3d!, raw=false); height=650, width=450, tellwidth=false, tellheight=false, alignmode=Outside())
    surfdiffscene = GLMakie.surface!(lscenediff, ts, xs, diffs[1])
    surfaxisdiff = lscenediff.scene.plots[1]
    surfaxisdiff.attributes.names.axisnames[] = ("t", "x", "Δu")
    surfaxisdiff.input_args[1][] = GeometryBasics.HyperRectangle{3,Float32}(Float32[-0.05, -0.05, -0.1], Float32[1.1, 2.1, 2.2])
    surfobsdiff = lscenediff.scene.plots[2].input_args[3]
    camscenediff = cam3d!(lscenediff.scene)

    colsize!(fig.layout, 2, Fixed(500))
    rowsize!(fig.layout, 2, Fixed(700))

    hm_bottom_sublayout = GridLayout()
    fig[3,1:2] = hm_bottom_sublayout
    hm_bottom_sublayout.height = 300
    colorbar = hm_bottom_sublayout[1, 1] = Colorbar(fig, surfscene; width=10)

    axiscontour = hm_bottom_sublayout[1, 2] = Axis(fig, title="u vs t-x")
    contourscene = GLMakie.contourf!(axiscontour, ts, xs, zs[1])
    axiscontour.xlabel="t"
    axiscontour.ylabel="x"
    contourplot = axiscontour.scene.plots[1]
    contourobs = contourplot.input_args[3]



    colorbar_diff = hm_bottom_sublayout[1, 4] = Colorbar(fig, surfdiffscene; width=10)

    axisdiffcontour = hm_bottom_sublayout[1, 3] = Axis(fig, title="abs(u-analytical) vs t-x")
    contourdiffscene = GLMakie.contourf!(axisdiffcontour, ts, xs, diffs[1])
    axisdiffcontour.xlabel="t"
    axisdiffcontour.ylabel="x"
    contourdiffplot = axisdiffcontour.scene.plots[1]
    contourdiffobs = contourdiffplot.input_args[3]


    title = fig[1,:] = Label(fig, "Learning Iteration: 0")
    rowsize!(fig.layout, 3, Fixed(200))
    rowsize!(fig.layout, 1, Fixed(50))


    focal_point = Vec3f0(0.5f0, 1.0f0, 0.0f0)
    vertical_offset = Vec3f0(0, 0, 1)
    #camscene = camera(lscene.scene)
    #camscenediff = camera(lscenediff.scene)

    i = 0
    cam_position = Vec3f0(focal_point .+ 1.2 .* Vec3f0(3 * cos(i/30), 4 * sin(i/30), 2.0 + 3.0 * sin(i/50)))
    up = Vec3f0(0,0,1)
    AbstractPlotting.update_cam!(lscene.scene, camscene, cam_position, focal_point, up)
    AbstractPlotting.update_cam!(lscenediff.scene, camscenediff, cam_position .+ vertical_offset, focal_point .+ vertical_offset)
    display(fig)
end

#i = 0
for i in 1:100
    #i += 1

    #AbstractPlotting.update_cam!(lscene.scene, camscene, cam_position, focal_point)
    #AbstractPlotting.update_cam!(lscenediff.scene, camscenediff, cam_position .+ vertical_offset, focal_point .+ vertical_offset)
    
    imod = ((1 * i - 1) % length(parameter_vectors)) + 1
    surfobs.val[:] .= @view zs[imod][:]
    contourobs.val[:] .= @view zs[imod][:]
    Observables.notify!(surfobs)
    Observables.notify!(contourobs)
    surfobsdiff.val[:] .= @view diffs[imod][:]
    contourdiffobs.val[:] .= @view diffs[imod][:]
    Observables.notify!(surfobsdiff)
    Observables.notify!(contourdiffobs)
    cam_position = Vec3f0(focal_point .+ 1.2 .* Vec3f0(3 * cos(i/30), 4 * sin(i/30), 2.0 + 3.0 * sin(i/50)))
    #camscene.eyeposition[] = Vec3f0(cam_position)
    lscene.scene.camera.view[] = AbstractPlotting.lookat(cam_position, focal_point, up)
    lscene.scene.camera.projectionview[] = lscene.scene.camera.projection[] * lscene.scene.camera.view[]
    #AbstractPlotting.update_cam!(lscene.scene, camscene)
    #camscenediff.eyeposition[] = Vec3f0(cam_position)
    lscenediff.scene.camera.view[] = AbstractPlotting.lookat(cam_position .+ vertical_offset, focal_point .+ vertical_offset, up)
    lscenediff.scene.camera.projectionview[] = lscenediff.scene.camera.projection[] * lscenediff.scene.camera.view[]
    title.elements[:text].text[] = "Learning Iteration: $(i * 10)"
    sleep(0.05)
    camdiff
end
fnty(camdiff)
save(datadir("hyperopt", experiment_title, "still_shots", experiment_params_str * "_$(i).jpg"), fig)
=#

function process_all_finished_experiments(experiment_title::AbstractString)
    #experiment_list_file = datadir("hyperopt", experiment_title, "experiment_list.csv")
    #experiment_list = Array(CSV.read(experiment_list_file, DataFrame; header=false))[:, 1]
    experiment_list = filter(s->occursin(r"ChainDims=", s), readdir(datadir("hyperopt", experiment_title)))
    
    finished_experiment_list = filter(s->is_experiment_finished(s, experiment_title), experiment_list)
    println("number of finished experiments to video: $(length(finished_experiment_list))")
    for experiment_str in finished_experiment_list

        #experiment_params = parse_savename_to_experiment(ExperimentParams, experiment_str)
        println("processing: $(experiment_str)")
        process_example_param(experiment_str)

    end


end

const experiment_title = "AdvectionOnly1DBigSearch"

process_all_finished_experiments(experiment_title)

#=
begin
    fig = MakieLayout.Figure(resolution=(1280,720), title="Iteration 0")


    lscene = LScene(fig[2,1], scenekw = (camera = cam3d!, raw=false))
    surfscene = GLMakie.surface!(lscene, ts, xs, zs[1])

    colorbar = fig[2, 2] = Colorbar(fig, surfscene)
    colorbar.width = 10
    axiscontour = fig[2, 3] = Axis(fig, title="u vs t-x contour")


    contourscene = GLMakie.contourf!(axiscontour, ts, xs, zs[1])
    axiscontour.xlabel="t"
    axiscontour.ylabel="x"

    title = fig[1,:] = Label(fig, "Learning Iteration: 0")

    display(fig)

    surfaxis = lscene.scene.plots[1]
    surfaxis.attributes.names.axisnames[] = ("t", "x", "u")
    surfaxis.input_args[1][] = GeometryBasics.HyperRectangle{3,Float32}(Float32[-0.05, -0.05, -1.1], Float32[1.1, 2.1, 2.2])
    surfobs = lscene.scene.plots[2].input_args[3]
    contourplot = axiscontour.scene.plots[1]
    contourobs = contourplot.input_args[3]

    focal_point = [0.5f0, 1.0f0, 0.0f0]
    for i in 1:30
        update_cam!(lscene.scene, cameracontrols(lscene.scene), Float32.(focal_point .+ [3 * cos(i/30), 4 * sin(i/30), 2.0 + 3.0 * sin(i/50)]), focal_point)
        imod = ((1 * i) % (length(parameter_vectors) - 1)) + 1
        surfobs.val[:] .= @view zs[imod][:]
        contourobs.val[:] .= @view zs[imod][:]
        Observables.notify!(surfobs)
        Observables.notify!(contourobs)
        title.elements[:text].text[] = "Learning Iteration: $(i)"
        sleep(0.01)
    end
end
=#

#fig = curfig()