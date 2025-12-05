# pem_mdc_analyze.jl 
#
# Usage:
#   julia --project pem_mdc_analyze.jl path_to_OUTROOT [delta_frac] [seed] [seed_end]
#   OUTROOT is output directory created by pem_mdc_train.jl
#   Examples:
#     julia --project pem_mdc_analyze.jl path_to_OUTROOT 0.01 2
#     julia --project pem_mdc_analyze.jl path_to_OUTROOT 0.01 1 10

include("pem_mdc_train.jl")   

using CSV, DataFrames, Statistics, Printf
using ComponentArrays
using Optimization, OptimizationOptimJL, Optim
using DifferentialEquations, SciMLSensitivity
using Plots


# Pure MSE (mean) from the already-saved predictions CSV of a seed
function baseline_mse_from_csv(run_dir::AbstractString)
    pred = CSV.read(joinpath(run_dir, "predictions_final.csv"), DataFrame)
    states = ["Never","Exposed","Presymptomatic","Symptomatic","Asymptomatic","Deaths"]
    diffs2 = Float64[]
    for s in states
        oc = Symbol(s*"_obs"); pc = Symbol(s*"_pred")
        if (oc in propertynames(pred)) && (pc in propertynames(pred))
            y  = collect(skipmissing(pred[!, oc]))
            ŷ  = collect(skipmissing(pred[!, pc]))
            n  = min(length(y), length(ŷ))
            append!(diffs2, (y[1:n] .- ŷ[1:n]).^2)
        end
    end
    @assert !isempty(diffs2) "No *_obs/*_pred pairs found in predictions_final.csv for $run_dir"
    return mean(diffs2)  # pure MSE, no regularizer
end

# Build observed data matrix (6×T) from the same CSV
function observed_matrix_from_csv(run_dir::AbstractString)
    pred = CSV.read(joinpath(run_dir, "predictions_final.csv"), DataFrame)
    states = ["Never","Exposed","Presymptomatic","Symptomatic","Asymptomatic","Deaths"]
    cols = Vector{Vector{Float64}}()
    for s in states
        oc = Symbol(s*"_obs")
        @assert oc in propertynames(pred) "Missing column $(oc) in predictions_final.csv"
        push!(cols, collect(skipmissing(pred[!, oc])))
    end
    # Stack rows as states (6×T)
    return reduce(hcat, cols)
end

# Robust normalized RMS curve distance with epsilon guard
curve_distance_norm(kappa_base::AbstractVector, kappa_tilde::AbstractVector) = begin
    @assert length(kappa_base) == length(kappa_tilde)
    w   = 1.0 / length(kappa_base)
    num = sqrt(sum(w .* (kappa_tilde .- kappa_base).^2))
    den = sqrt(sum(w .* (kappa_base.^2))) + 1e-12
    num / den
end

# Lean ODE solve (fix to test? memory-friendly, need to check if it works well on macOS?)
lean_solve(prob, tdata) = solve(prob, Tsit5();
    saveat          = tdata,
    save_everystep  = false,
    save_start      = false,
    save_end        = true,
    dense           = false,
    save_idxs       = 1:6,
    sensealg        = QuadratureAdjoint(autojacvec = ZygoteVJP())
)

# MDC loss: PURE MSE (mean), using lean solve
function pem_loss_md(p_var; prob_md, tdata, data_mat)
    sol  = lean_solve(remake(prob_md, p=p_var), tdata)
    pred = Array(sol)  # 6×T
    L    = mean((data_mat .- pred).^2)
    sol = nothing; pred = nothing
    return L
end

# Obtain baseline κ grid robustly (md.κ_grid or md.kappa_grid)
function baseline_kappa_grid(md)
    if hasproperty(md, Symbol("κ_grid"))
        return getfield(md, Symbol("κ_grid"))
    elseif hasproperty(md, :kappa_grid)
        return getfield(md, :kappa_grid)
    else
        error("md.κ_grid (or md.kappa_grid) not found in MDC context")
    end
end

# MDC runner 
"""
    run_mdcurve_for_seed(seed; outroot, delta_frac=0.01, λ=1.0, maxiters=7000)

Run the minimally disruptive κ test **without retraining**, writing:
  - run_<seed>/mdcurve_summary.csv
  - run_<seed>/mdcurve_kappa_overlay.png
"""
function run_mdcurve_for_seed(seed::Integer; outroot::AbstractString,
                              delta_frac::Float64=0.01, λ::Float64=1.0, maxiters::Int=7000)

    run_dir = joinpath(outroot, "run_$(seed)")
    @assert isdir(run_dir) "Run directory not found: $run_dir"

    # Baseline pure-MSE loss and budget
    L_star_mse = baseline_mse_from_csv(run_dir)
    L_budget   = (1 + delta_frac) * L_star_mse
    @info "[MDC seed=$(seed)] baseline MSE = $(L_star_mse), δ=$(round(100*delta_frac,digits=2))%, budget = $(L_budget)"

    # Build MDC context via your existing code
    built  = build_run(seed)                                  # from your file
    md     = build_mdcurve(seed, run_dir, built; delta_frac=delta_frac)
    prob_md, tdata = md.prob_md, md.tdata
    data_mat = observed_matrix_from_csv(run_dir)              # 6×T
    kbase    = baseline_kappa_grid(md)

    # Budget-centered objective
    function obj(u, p)
        p_var = ComponentArray(u, getaxes(md.p_var0))
        L_alt = pem_loss_md(p_var; prob_md=prob_md, tdata=tdata, data_mat=data_mat)
        κ̃    = md.kappa_tilde_on_grid(p_var)
        D     = curve_distance_norm(kbase, κ̃)
        J     = (L_alt - L_budget)^2 - λ * D
        return J
    end

    optf  = Optimization.OptimizationFunction(obj, Optimization.AutoZygote())
    prob  = Optimization.OptimizationProblem(optf, md.p0_vec, nothing)
    res   = Optimization.solve(prob, LBFGS(linesearch=BackTracking());
                               maxiters=maxiters, store_trace=false, show_trace=false)

    # Final metrics
    p_opt = ComponentArray(res.u, getaxes(md.p_var0))
    L_alt = pem_loss_md(p_opt; prob_md=prob_md, tdata=tdata, data_mat=data_mat)
    κ̃    = md.kappa_tilde_on_grid(p_opt)
    dist  = curve_distance_norm(kbase, κ̃)
    Δloss = 100 * (L_alt - L_star_mse) / max(L_star_mse, 1e-12)

    # Warn if we drifted too far from the budget shell
    if abs(L_alt - L_budget) / max(L_budget, 1e-12) > 0.02
        @warn "[MDC seed=$(seed)] Final loss off budget (>2% of L*): L_alt=$(L_alt), L_budget=$(L_budget)"
    end

    # Overlay plot (close figure to free memory)
    plt = plot(tdata, kbase, label="κ baseline")
    plot!(plt, tdata, κ̃, label="κ̃ (min-disrupt)", linestyle=:dash)
    ann = @sprintf "(MSE) Δloss ≈ %.2f%% | Δκ_norm = %.4f", Δloss, dist
    annotate!(plt, (tdata[1], maximum(vcat(kbase, κ̃))), text(ann, 8, :left))
    xlabel!("time"); ylabel!("κ"); title!("Seed $seed – minimally disruptive κ")
    savefig(plt, joinpath(run_dir, "mdcurve_kappa_overlay.png"))
    close(plt); try; import GR; GR.clearws(); catch; end

    # Consistent summary CSV
    CSV.write(joinpath(run_dir, "mdcurve_summary.csv"),
              DataFrame(; seed=seed,
                         L_star=L_star_mse, L_alt=L_alt,
                         rel_dloss_percent=Δloss,
                         norm_curve_distance=dist))
    @info "[MDC seed=$(seed)] Δloss%=$(round(Δloss,digits=3)), Δκ_norm=$(round(dist,digits=6))"
    return nothing
end

# CLI / Main guard 

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) == 0
        println("Usage: julia --project pem_mdc_train.jl OUTROOT [delta_frac] [seed] [seed_end]")
        println("Examples:")
        println("  julia --project pem_mdc_train.jl path_to_OUTROOT")
        println("  julia --project pem_mdc_train.jl path_to_OUTROOT 0.01 2")
        println("  julia --project pem_mdc_train.jl path_to_OUTROOT 0.01 1 10")
        exit(1)
    end
    outroot    = ARGS[1]
    delta_frac = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 0.01
    if length(ARGS) >= 4
        s1 = parse(Int, ARGS[3]); s2 = parse(Int, ARGS[4])
        for s in s1:s2
            try
                run_mdcurve_for_seed(s; outroot=outroot, delta_frac=delta_frac)
            catch e
                @warn "MDC failed for seed $s" exception=(e, catch_backtrace())
            end
        end
    else
        seed = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1
        run_mdcurve_for_seed(seed; outroot=outroot, delta_frac=delta_frac)
    end
end
