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
#using DifferentialEquations, SciMLSensitivity
using OrdinaryDiffEq, SciMLSensitivity
using Plots
import GR


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
# CLI 
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
