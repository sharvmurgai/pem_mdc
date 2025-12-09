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
vm@vm-Inspiron-15-3511:~/abm_ude$ cat pem_mdc_train.jl
# pem_mdc_train.jl
#
# Example usage:
# $ julia --project=. pem_mdc_train.jl
#
cd(dirname(@__FILE__))

using Pkg
Pkg.activate("julia_test")

using MAT, JLD2
using CSV, DataFrames
using Random, StableRNGs, Statistics, LinearAlgebra
BLAS.set_num_threads(1)
using Colors
using Dates
using OrdinaryDiffEq, ModelingToolkit
using SciMLSensitivity, SciMLBase
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using ComponentArrays
using Lux, Zygote, StaticArrays
using Plots, Measures
using LineSearches
using Base.Threads   # Emsemble changes

# Include utilities
include("utils_1NN.jl")   

# Config
const MODEL_NAME     = "SEInsIsIaDR"

# use ths for smoke test
#const ADAM_maxiters  = 15
#const LBFGS_maxiters = 5
#const N_RUNS         = 5 

const ADAM_maxiters  = 1500
const LBFGS_maxiters = 500
const N_RUNS         = 1 # change to 10 ensemble

const SCRIPT_NAME = splitext(basename(PROGRAM_FILE))[1]
const OUTROOT_BASE = SCRIPT_NAME
const OUTROOT = OUTROOT_BASE * "_" * Dates.format(Dates.now(), "yyyy-mm-dd_HHMMSS")


# ONLY κ and K_obs are learned
const p_true = (T_E=5.0, T_In=7.0, T_i=10.0, eta_a=0.5, p_trans=3.7/100.0,
                fa=0.35, fr=0.90)

# States: 6 observed + 1 recovered (R) internally
const STATE_NAMES = ["Never","Exposed","Presymptomatic","Symptomatic","Asymptomatic","Deaths"]
const S = length(STATE_NAMES)

const OBS_MASK = ones(Float64, S)

# Training regularizer. Through various runs, we found 10.0 gives the best MSE.
const REG_LAMBDA = 10.0

# Data
df = DataFrame(CSV.File("avg_output.dat", delim=' ', ignorerepeated=true))
selected_df, tdata, tspan, train_n_days, data_norm, u0_norm, max_vals = process_data(df, MODEL_NAME)

# Denormalize back to original scale for mechanistic ODE fitting
data = data_norm .* max_vals            # 6×T
u0_6 = u0_norm   .* max_vals            # 6

# Recovered initial guess (from df if available; else 0)
R0_guess = hasproperty(df, :Immune) ? Float64(df[1, :Immune]) : 0.0
N0 = sum(u0_6) + R0_guess  # total incl. R, excl. D in denominator where needed
R0 = max(R0_guess, (N0 - u0_6[end]) - sum(u0_6[1:5]))  # ensure nonnegative
u0 = vcat(u0_6, R0)  

# Interpolator y(t) for the observed 6 states (original scale)
function make_linear_interpolator(tgrid::AbstractVector, Y::AbstractMatrix)
    cols = [@view Y[:, i] for i in 1:length(tgrid)]
    function y_of_t(t::Real)
        if t <= tgrid[1]; return copy(cols[1]); end
        if t >= tgrid[end]; return copy(cols[end]); end
        i = clamp(searchsortedlast(tgrid, t), 1, length(tgrid)-1)
        t0 = tgrid[i]; t1 = tgrid[i+1]; w = (t - t0) / (t1 - t0)
        @. (1 - w) * cols[i] + w * cols[i+1]
    end
    return y_of_t
end
const y_of_t = make_linear_interpolator(tdata, data)  # 6-vector

function save_predictions_csv(sol_arr::AbstractArray, tdata::AbstractVector, data_arr::AbstractArray;
                              outdir::AbstractString = "PEM_UDE_KAPPA_OG__csv", tag::AbstractString = "")
    mkpath(outdir)
    cols = Dict{Symbol, Any}()
    cols[:time] = collect(tdata)
    @inbounds for (i, nm) in enumerate(STATE_NAMES)
         cols[Symbol("$(nm)_obs")]  = vec(data_arr[i, :])
        cols[Symbol("$(nm)_pred")] = vec(sol_arr[i, :])
    end
    outpath = joinpath(outdir, isempty(tag) ? "predictions.csv" : "predictions_$(tag).csv")
    CSV.write(outpath, DataFrame(cols))
end

# Simple R-square on flattened observed states
function r2_score(y_true::AbstractArray, y_pred::AbstractArray)
    μ = mean(y_true)
    ss_res = sum((y_true .- y_pred).^2)
    ss_tot = sum((y_true .- μ).^2)
    return ss_tot ≈ 0 ? NaN : 1 - (ss_res / ss_tot)
end

# Metric container
struct FitMetrics
    mse_overall::Float64
    mae_overall::Float64
    r2_overall::Float64
    mse_per_state::Vector{Float64}
    seed::Int
    adam_loss::Float64
    lbfgs_loss::Float64
    elapsed_s::Float64
    retcode::Any
end

# Build a fresh run (network + params + ODEProblem) bound to a given seed
softplus(x) = log1p(exp(x))

function build_run(seed::Int;
                   model::String=MODEL_NAME,
                   p_true::NamedTuple=p_true,
                   obs_mask::AbstractVector{<:Real}=OBS_MASK,
                   tdata::AbstractVector=tdata,
                   data::AbstractMatrix=data,
                   u0::AbstractVector=u0,
                   N0::Real=N0)

    rng = StableRNGs.StableRNG(seed)

    # κ(u;θ) network
    nn_kappa_local = Chain(
        Dense(6, 32, tanh),
        Dense(32, 16, tanh),
        Dense(16, 1)
    )
    pκ_init_local, stκ_local = Lux.setup(rng, nn_kappa_local)

    # Trainable per-state nudging gains for 6 observed states
    Kraw_init_local = fill(-2.0, S)  # softplus -> small positive

    # Pack parameters
    p0_local = ComponentArray(; θκ=pκ_init_local, Kraw=Kraw_init_local)

    # RHS (PEM nudging on first 6 states)
    function rhs!(du, u, p, t)
        T_E   = p_true.T_E    ; T_In  = p_true.T_In
        T_i   = p_true.T_i    ; eta_a = p_true.eta_a
        p_tr  = p_true.p_trans
        fa    = p_true.fa     ; fr    = p_true.fr

        S, E, Ins, Is, Ia, D, R = u

        # κ(u;θ)
        NminusD = max(S + E + Ins + Is + Ia + R, 1e-9)
	    xκ = SVector{6,Float64}( E/NminusD, Ins/NminusD, Is/NminusD, Ia/NminusD, D/max(N0,1e-9), R/NminusD)

        κ = softplus(nn_kappa_local(xκ, p.θκ, stκ_local)[1][1]) + 1e-8

        # Force of infection
        λ = p_tr * κ * (eta_a*Ins + Is + eta_a*Ia) / max((S + E + Ins + Is + Ia + R), 1e-9)

        T_ins = T_In - T_E
        T_s   = T_E + T_i - T_In

        du[1] = -λ * S
        du[2] =  λ * S - E / T_E
        du[3] = (1.0 - fa) * E / T_E - Ins / T_ins
        du[4] =  Ins / T_ins - Is / T_s
        du[5] =  fa * E / T_E - Ia / T_i
        du[6] = (1.0 - fr) * Is / T_s
        du[7] =  fr * Is / T_s + Ia / T_i

        # PEM nudging on observed states 1..6
        y = y_of_t(t)  # 6-vector
        @inbounds for i in 1:6
            Ki = clamp(softplus(p.Kraw[i]), 0.0, 10.0)
            du[i] += (obs_mask[i] * Ki) * (y[i] - u[i])
        end
        return nothing
    end

    prob_local = ODEProblem((du,u,p,t)->rhs!(du,u,p,t),
                             u0, (tdata[1], tdata[end]), p0_local;
                             saveat=tdata)

    # Loss on observed states only (1..6)
    obs_idx = 1:6
    function loss_fn_local(pvec)
        p_struct = ComponentArray(pvec, getaxes(p0_local))
        sol = solve(remake(prob_local, p=p_struct), Tsit5(); saveat=tdata)
        if !SciMLBase.successful_retcode(sol.retcode); return Inf; end
        pred = Array(sol)[obs_idx, :]
        
        mse_loss = mean(abs2.(data .- pred))
        reg_loss = REG_LAMBDA * sum(softplus.(p_struct.Kraw))
        return mse_loss + reg_loss
    end

    return (; prob_local, p0_local, loss_fn_local, nn_kappa_local, stκ_local)
end

# One training pass (ADAM -> LBFGS) for a given seed; saves outputs; returns metrics
function train_once(seed::Int; outroot::String=OUTROOT)
    run_dir = joinpath(outroot, "run_$(seed)")
    mkpath(run_dir)

    # Build per-seed run
    built = build_run(seed)
    prob_local = built.prob_local
    p0_local   = built.p0_local
    loss_fn_local = built.loss_fn_local

    adtype = Optimization.AutoZygote()
    optf   = Optimization.OptimizationFunction((x,_)->loss_fn_local(x), adtype)
    optprob = Optimization.OptimizationProblem(optf, getdata(p0_local))

    # Loss history & callback (covers ADAM + LBFGS)
    loss_hist = Float64[]
    callback = function (p, l)
        push!(loss_hist, l)
        if length(loss_hist) % 100 == 0
            println("[$seed] Iter: $(length(loss_hist)) \t Loss: $(l)")
        end
        return false
    end

    t0 = time()

    # ADAM 
    println("[$seed] Starting ADAM optimization (maxiters = $(ADAM_maxiters))...")
    res_adam = Optimization.solve(optprob, ADAM(1e-3); callback=callback, maxiters=ADAM_maxiters)
    adam_loss = res_adam.minimum
    println("[$seed] ADAM finished. Loss = $(adam_loss)")

    # LBFGS 
    p_LBFGS_init = getdata(ComponentArray(res_adam.u, getaxes(p0_local)))
    optprob_LBFGS = Optimization.OptimizationProblem(optf, p_LBFGS_init)
    
    println("[$seed] Starting L-BFGS optimization (maxiters = $(LBFGS_maxiters))...")
    res_lbfgs = Optimization.solve(optprob_LBFGS, LBFGS(linesearch=BackTracking());
                                   callback=callback, maxiters=LBFGS_maxiters)
    lbfgs_loss = res_lbfgs.minimum
    elapsed = time() - t0
    println("[$seed] L-BFGS finished. Final Loss = $(lbfgs_loss). Elapsed: $(round(elapsed, digits=2))s")


    # Save loss history (CSV only; we will plot later)
    if !isempty(loss_hist)
        loss_df = DataFrame(iter = 1:length(loss_hist), loss = loss_hist)
        CSV.write(joinpath(run_dir, "loss_history.csv"), loss_df)
    end

    # Final solution & predictions
    println("[$seed] Saving final predictions and metrics...")
    p_final = ComponentArray(res_lbfgs.u, getaxes(p0_local))
    sol_final = solve(remake(prob_local, p=p_final), Tsit5(); saveat=tdata)
    pred_final = Array(sol_final)[1:6, :]

    # (MDC): save baseline checkpoint for identifiability
    #JLD2.@save joinpath(run_dir, "baseline_checkpoint.jld2") p_final sol_final L_star=lbfgs_loss
    JLD2.@save joinpath(run_dir, "baseline_checkpoint.jld2") p_final L_star=lbfgs_loss u0=prob_local.u0 tspan=prob_local.tspan

    # Save per-run predictions
    save_predictions_csv(pred_final, tdata, data; outdir=run_dir, tag="final")

    # Compute and save metrics
    y_true = data
    y_pred = pred_final
    mse_per_state = [mean((y_true[i, :] .- y_pred[i, :]).^2) for i in 1:size(y_true,1)]
    mae_overall = mean(abs.(y_true .- y_pred))
    mse_overall = mean((y_true .- y_pred).^2)
    r2_overall  = r2_score(vec(y_true), vec(y_pred))

    CSV.write(joinpath(run_dir, "metrics.csv"),
        DataFrame(; seed=seed,
                    adam_loss=adam_loss,
                    lbfgs_loss=lbfgs_loss,
                    mse_overall=mse_overall,
                    mae_overall=mae_overall,
                    r2_overall=r2_overall,
                    mse_Never=mse_per_state[1],
                    mse_Exposed=mse_per_state[2],
                    mse_Presymptomatic=mse_per_state[3],
                    mse_Symptomatic=mse_per_state[4],
                    mse_Asymptomatic=mse_per_state[5],
                    mse_Deaths=mse_per_state[6],
                    elapsed_s=elapsed,
                    retcode=string(sol_final.retcode)
        )
    )
    
    println("[$seed] Metrics saved.")

    return FitMetrics(mse_overall, mae_overall, r2_overall, mse_per_state, seed,
                      adam_loss, lbfgs_loss, elapsed, sol_final.retcode)
end

# Ensemble loop (seeds 1..N_RUNS) + aggregation
function run_ensemble(; N::Int=N_RUNS, outroot::String=OUTROOT)
    mkpath(outroot)
    results = Vector{FitMetrics}(undef, N)

    @threads for i in 1:N
        seed = i
        try
            println("##### Starting run $seed on thread $(threadid()) at $(Dates.now())")
            results[i] = train_once(seed; outroot=outroot)  # NO plotting in threads
            println("##### FINISHED run $seed on thread $(threadid()) at $(Dates.now())")
        catch e
            @warn "Run $seed failed" error=e
            results[i] = FitMetrics(NaN, NaN, NaN, fill(NaN, 6), seed, NaN, NaN, NaN, :Failure)
        end
    end

    println("[Ensemble] All runs complete. Aggregating summary table...")
    
    # Aggregate to summary table
    df_sum = DataFrame(
        seed = [r.seed for r in results],
        adam_loss = [r.adam_loss for r in results],
        lbfgs_loss = [r.lbfgs_loss for r in results],
        mse_overall = [r.mse_overall for r in results],
        mae_overall = [r.mae_overall for r in results],
        r2_overall  = [r.r2_overall for r in results],
        mse_Never   = [r.mse_per_state[1] for r in results],
        mse_Exposed = [r.mse_per_state[2] for r in results],
        mse_Presymptomatic = [r.mse_per_state[3] for r in results],
        mse_Symptomatic    = [r.mse_per_state[4] for r in results],
        mse_Asymptomatic   = [r.mse_per_state[5] for r in results],
        mse_Deaths         = [r.mse_per_state[6] for r in results],
        elapsed_s   = [r.elapsed_s for r in results],
        retcode     = [string(r.retcode) for r in results]
    )

    CSV.write(joinpath(outroot, "ensemble_summary.csv"), df_sum)

    # Robust summary over successful runs
    good_df = filter(:mse_overall => isfinite, df_sum)
    Ngood   = nrow(good_df)

    if Ngood == 0
        println("=== Ensemble: no successful runs ===")
        return df_sum
    end

    μ_mse = mean(good_df.mse_overall)
    σ_mse = std(good_df.mse_overall)
    μ_r2  = mean(good_df.r2_overall)
    σ_r2  = std(good_df.r2_overall)

    println("=== Ensemble (N=$Ngood / $N) ===")
    println("Overall MSE: $(round(μ_mse, sigdigits=5)) ± $(round(σ_mse, sigdigits=5))")
    println("Overall R² : $(round(μ_r2,  sigdigits=5)) ± $(round(σ_r2,  sigdigits=5))")

    return df_sum
end

# Post-processing: generate plots from CSVs (sequential, safe for GR)
function generate_plots(; outroot::String=OUTROOT, N::Int=N_RUNS)
    println("[INFO] Generating plots from CSVs in $outroot")

    # Per-seed plots
    for seed in 1:N
        run_dir = joinpath(outroot, "run_$(seed)")

        # Obs vs pred per state
        pred_path = joinpath(run_dir, "predictions_final.csv")
        if isfile(pred_path)
            df = CSV.read(pred_path, DataFrame)
            
            # FINAL PLOTTING FIX 
            # Get column names as Strings
            names_df = names(df)

            # Find all "_obs" columns
            for obs_col_str in names_df
                if endswith(obs_col_str, "_obs")
                    
                    # Reconstruct the state name and pred_col name as Strings
                    state = replace(obs_col_str, "_obs" => "")
                    pred_col_str = "$(state)_pred"

                    # Check if the matching "pred" column (as a String) also exists
                    if !(pred_col_str in names_df)
                        @warn "Missing matching column for $obs_col_str"
                        continue
                    end

                    # Now plot (using Strings to index is fine)
                    plt = plot(df.time, df[!, obs_col_str],
                               label  = "obs",
                               xlabel = "time",
                               ylabel = "count",
                               title  = "Seed $seed – $state")
                    plot!(plt, df.time, df[!, pred_col_str], label="pred")
                    savefig(plt, joinpath(run_dir, "state_$(state)_obs_vs_pred.png"))
                end
            end
            # END FINAL PLOTTING FIX 
        end

        # Loss curve per seed (assumes loss_history.csv exists)
        loss_path = joinpath(run_dir, "loss_history.csv")
        if isfile(loss_path)
            ldf = CSV.read(loss_path, DataFrame)
            cols = names(ldf)
            has_iter = (:iter in cols) || ("iter" in cols)
            has_loss = (:loss in cols) || ("loss" in cols)
            if has_iter && has_loss
                iter_col = :iter in cols ? :iter : Symbol("iter")
                loss_col = :loss in cols ? :loss : Symbol("loss")
                plt_loss = plot(ldf[!, iter_col], ldf[!, loss_col],
                                xlabel = "Iteration",
                                ylabel = "Loss",
                                yscale = :log10,
                                title  = "Loss curve (seed = $seed)")
                savefig(plt_loss, joinpath(run_dir, "loss_curve.png"))
            end
        end
    end

    # Across-seed "loss per state" from ensemble_summary.csv
    sum_path = joinpath(outroot, "ensemble_summary.csv")
    if isfile(sum_path)
        df_sum = CSV.read(sum_path, DataFrame)
        cols_sum = names(df_sum)

        if (:seed in cols_sum) && (:mse_overall in cols_sum)
            plt_overall = plot(df_sum[!, :seed], df_sum[!, :mse_overall],
                               seriestype = :scatter,
                               xlabel = "seed",
                               ylabel = "MSE (overall)",
                               title  = "Overall MSE across seeds")
            savefig(plt_overall, joinpath(outroot, "overall_mse_across_seeds.png"))
        end

        for state in STATE_NAMES
            col = Symbol("mse_$(state)")
            if col in cols_sum
                plt_state = plot(df_sum[!, :seed], df_sum[!, col],
                                 seriestype = :scatter,
                                 xlabel = "seed",
                                 ylabel = "MSE",
                                 title  = "MSE for $state across seeds")
                savefig(plt_state, joinpath(outroot, "mse_$(state)_across_seeds.png"))
            end
        end
    end
end

# Minimally Disruptive Curve (MDC) identifiability check
#
# Stable inverse-softplus
@inline invsoftplus(y::Real) = log(exp(y) - 1 + 1e-12)

# Build κ-feature vector (mirror of rhs!)
@inline function kappa_features(u::AbstractVector, N0::Real)
    S,E,Ins,Is,Ia,D,R = u
    NminusD = max(S + E + Ins + Is + Ia + R, 1e-9)
    return [E/NminusD, Ins/NminusD, Is/NminusD, Ia/NminusD, D/max(N0,1e-9), R/NminusD]
end

# Tiny bounded perturbation head Δ(x; φ): κ̃ = κ * exp(Δ), preserves positivity
function make_delta_head(rng::AbstractRNG)
    nnΔ = Chain(Dense(6, 8, tanh), Dense(8, 1))
    pΔ, stΔ = Lux.setup(rng, nnΔ)
    αraw0 = -2.0                         # α = softplus(αraw) > 0, start small
    p_md0 = ComponentArray(; θΔ=pΔ, αraw=αraw0)
    return nnΔ, stΔ, p_md0
end

# build_mdcurve (MSE-consistent budget) 
function build_mdcurve(seed::Int, run_dir::String, built; delta_frac::Float64=0.01)
    ckpt_path = joinpath(run_dir, "baseline_checkpoint.jld2")
    @assert isfile(ckpt_path) "Missing baseline_checkpoint.jld2 in $run_dir. Run training for this seed first."
    #
    # Error 
    # JLD2.@load ckpt_path p_final sol_final L_star   # L_star = training loss incl. regularizer  (kept for reference)  [see train_once save]  
    # Compute consistent baseline MSE from the saved baseline solution (pred vs data):
    # use MSE-only as the baseline for MDC budget & Δloss%
    #pred0 = Array(sol_final)[1:6, :]


    # NEW WORKING CODE
    JLD2.@load ckpt_path p_final L_star # removed sol_final
    prob_base = remake(built.prob_local, p=p_final)
    sol_final = solve(prob_base, Tsit5(); saveat=built.prob_local.kwargs[:saveat])

    pred0 = Array(sol_final)[1:6, :]


    L_star_mse = mean(abs2.(data .- pred0))

    nnκ  = built.nn_kappa_local
    stκ  = built.stκ_local

    # Δ-head
    rng = StableRNGs.StableRNG(seed + 10_000)
    nnΔ, stΔ, p_md0 = make_delta_head(rng)

    # Build grid from baseline trajectory times
    sol_arr = Array(sol_final)  # 7×T
    T = size(sol_arr, 2)

    Xgrid = Vector{Vector{Float64}}(undef, T)
    for j in 1:T
        u_sv = SVector{7,Float64}(sol_arr[:, j])
        Xgrid[j] = kappa_features(u_sv, N0)
    end
    w = fill(1.0 / T, T)

    # Baseline κ on grid 
    function kappa_base(x::Vector{Float64})
        κ_raw = nnκ(x, p_final.θκ, stκ)[1][1]
        return softplus(κ_raw) + 1e-8
    end
    κ_grid = [kappa_base(x) for x in Xgrid]

    # κ̃(x;φ) on grid 
    function kappa_tilde_on_grid(p_md::ComponentArray)
        α = softplus(p_md.αraw)
        κ̃ = [begin
            Δ = α * tanh(nnΔ(Xgrid[j], p_md.θΔ, stΔ)[1][1])
            κ_grid[j] * exp(Δ)
        end for j in 1:T]
        return κ̃
    end

    # Normalized RMS curve distance on the grid 
    κ_rms = sqrt(sum(w[j] * κ_grid[j]^2 for j in 1:T))
    function curve_distance_norm(p_md::ComponentArray)
        κ̃ = kappa_tilde_on_grid(p_md)
        num = sqrt(sum(w[j] * (κ̃[j] - κ_grid[j])^2 for j in 1:T))
        return κ_rms > 0 ? num / κ_rms : 0.0
    end

    # MDC ODE (κ -> κ̃, same nudging K)
    function rhs_md!(du, u, p, t)
        T_E = p_true.T_E; T_In = p_true.T_In; T_i = p_true.T_i
        eta_a = p_true.eta_a; p_tr = p_true.p_trans
        fa = p_true.fa; fr = p_true.fr

        S,E,Ins,Is,Ia,D,R = u
        xκ_sv = SVector(S, E, Ins, Is, Ia, D, R)
        xκ = kappa_features(xκ_sv, N0)

        κb = softplus(nnκ(xκ, p_final.θκ, stκ)[1][1]) + 1e-8
        α  = softplus(p.αraw)
        Δ  = α * tanh(nnΔ(xκ, p.θΔ, stΔ)[1][1])
        κ  = κb * exp(Δ)

        λ = p_tr * κ * (eta_a*Ins + Is + eta_a*Ia) / max((S + E + Ins + Is + Ia + R), 1e-9)

        T_ins = T_In - T_E
        T_s   = T_E + T_i - T_In

        du[1] = -λ * S
        du[2] =  λ * S - E / T_E
        du[3] = (1.0 - fa) * E / T_E - Ins / T_ins
        du[4] =  Ins / T_ins - Is / T_s
        du[5] =  fa * E / T_E - Ia / T_i
        du[6] = (1.0 - fr) * Is / T_s
        du[7] =  fr * Is / T_s + Ia / T_i

        y = y_of_t(t)
        @inbounds for i in 1:6
            Ki = clamp(softplus(p_final.Kraw[i]), 0.0, 10.0)
            du[i] += (OBS_MASK[i] * Ki) * (y[i] - u[i])
        end
        return nothing
    end

    prob_md = ODEProblem((du,u,p,t)->rhs_md!(du,u,p,t),
                         copy(sol_final.u[1]), (tdata[1], tdata[end]), nothing;
                         saveat=tdata)

    # Optimize only Δ-head params (unchanged)
    p_var0 = ComponentArray(make_delta_head(StableRNGs.StableRNG(seed + 12345))[3])

    # MDC loss uses MSE-only, now consistent with budget
    function pem_loss_md(p_vec)
        p_var = ComponentArray(p_vec, getaxes(p_var0))
        sol = solve(remake(prob_md, p=p_var), Tsit5(); saveat=tdata)
        if !SciMLBase.successful_retcode(sol.retcode); return Inf; end
        pred = Array(sol)[1:6, :]
        return mean(abs2.(data .- pred))
    end

    # Build the budget from MSE-only baseline
    L_budget = (1 + delta_frac) * L_star_mse

    # Correct MDC objective 
    function objective(p_vec; λ=1.0, μ=200.0)
        L = pem_loss_md(p_vec)
        D = curve_distance_norm(ComponentArray(p_vec, getaxes(p_var0)))
        gap = L - L_budget
        return gap^2 - λ*D, L, D
    end

    # also return L_star_mse for downstream CSV and Δloss% calculations
    #return (; objective, p_var0, κ_grid, kappa_tilde_on_grid, L_star, L_star_mse, L_budget)
    return (; objective, p_var0, κ_grid, kappa_tilde_on_grid, L_star, L_star_mse, L_budget, prob_md, tdata)
end

# Construct the MDC problem for a seed using its baseline checkpoint
function build_mdcurve_old(seed::Int, run_dir::String, built; delta_frac::Float64=0.01)
    ckpt_path = joinpath(run_dir, "baseline_checkpoint.jld2")
    @assert isfile(ckpt_path) "Missing baseline_checkpoint.jld2 in $run_dir. Run training for this seed first."
    JLD2.@load ckpt_path p_final sol_final L_star

    nnκ  = built.nn_kappa_local
    stκ  = built.stκ_local

    # Δ-head
    rng = StableRNGs.StableRNG(seed + 10_000)
    nnΔ, stΔ, p_md0 = make_delta_head(rng)

    # Build grid from baseline trajectory times
    sol_arr = Array(sol_final)  # 7×T
    T = size(sol_arr, 2)
    
    Xgrid = Vector{Vector{Float64}}(undef, T)
    for j in 1:T
        u_sv = SVector{7,Float64}(sol_arr[:, j])
        Xgrid[j] = kappa_features(u_sv, N0) # This returns Vector{Float64}
    end
    w = fill(1.0 / T, T)

    # Baseline κ on grid
    function kappa_base(x::Vector{Float64})
        κ_raw = nnκ(x, p_final.θκ, stκ)[1][1]
        return softplus(κ_raw) + 1e-8
    end
    κ_grid = [kappa_base(x) for x in Xgrid]

    # κ̃(x;φ) on grid
    function kappa_tilde_on_grid(p_md::ComponentArray)
        α = softplus(p_md.αraw)

        κ̃ = [begin
            Δ = α * tanh(nnΔ(Xgrid[j], p_md.θΔ, stΔ)[1][1])
            κ_grid[j] * exp(Δ)
        end for j in 1:T]

        return κ̃
    end

    # Normalized RMS curve distance on the grid
    κ_rms = sqrt(sum(w[j] * κ_grid[j]^2 for j in 1:T))
    function curve_distance_norm(p_md::ComponentArray)
        κ̃ = kappa_tilde_on_grid(p_md)
        num = sqrt(sum(w[j] * (κ̃[j] - κ_grid[j])^2 for j in 1:T))
        return κ_rms > 0 ? num / κ_rms : 0.0
    end

    # ODE with κ replaced by κ̃ = κ * exp(Δ); same PEM nudging
    function rhs_md!(du, u, p, t)
        T_E = p_true.T_E; T_In = p_true.T_In; T_i = p_true.T_i
        eta_a = p_true.eta_a; p_tr = p_true.p_trans
        fa = p_true.fa; fr = p_true.fr

        S,E,Ins,Is,Ia,D,R = u
        
	    xκ_sv = SVector(S, E, Ins, Is, Ia, D, R)
        xκ = kappa_features(xκ_sv, N0)

        κb = softplus(nnκ(xκ, p_final.θκ, stκ)[1][1]) + 1e-8
        α  = softplus(p.αraw)
        Δ  = α * tanh(nnΔ(xκ, p.θΔ, stΔ)[1][1])
        κ  = κb * exp(Δ)

        λ = p_tr * κ * (eta_a*Ins + Is + eta_a*Ia) / max((S + E + Ins + Is + Ia + R), 1e-9)

        T_ins = T_In - T_E
        T_s   = T_E + T_i - T_In

        du[1] = -λ * S
        du[2] =  λ * S - E / T_E
        du[3] = (1.0 - fa) * E / T_E - Ins / T_ins
        du[4] =  Ins / T_ins - Is / T_s
        du[5] =  fa * E / T_E - Ia / T_i
        du[6] = (1.0 - fr) * Is / T_s
        du[7] =  fr * Is / T_s + Ia / T_i

        y = y_of_t(t)
        @inbounds for i in 1:6
            Ki = clamp(softplus(p_final.Kraw[i]), 0.0, 10.0)
            du[i] += (OBS_MASK[i] * Ki) * (y[i] - u[i])
        end
        return nothing
    end

    prob_md = ODEProblem((du,u,p,t)->rhs_md!(du,u,p,t),
                         copy(sol_final.u[1]), (tdata[1], tdata[end]), nothing;
                         saveat=tdata)

    # Variables to optimize: only Δ-head params (θΔ, αraw)
    p_var0 = ComponentArray(make_delta_head(StableRNGs.StableRNG(seed + 12345))[3])

    # PEM loss using κ̃
    function pem_loss_md(p_vec)
        p_var = ComponentArray(p_vec, getaxes(p_var0))
        sol = solve(remake(prob_md, p=p_var), Tsit5(); saveat=tdata)
        if !SciMLBase.successful_retcode(sol.retcode); return Inf; end
        pred = Array(sol)[1:6, :]
        return mean(abs2.(data .- pred))
    end

    # Implement the correct MDC objective 
    # The new objective (gap^2 - λ*D) forces the optimizer to
    # stay AT the loss budget (minimizing gap^2) and spend its
    # effort maximizing the distance (by minimizing -λ*D).
    L_budget = (1 + delta_frac) * L_star
    function objective(p_vec; λ=1.0, μ=200.0) # μ is no longer used
        L = pem_loss_md(p_vec)
        D = curve_distance_norm(ComponentArray(p_vec, getaxes(p_var0)))
        
        # old, wrong OBJECTIVE:
        # pen = μ * max(0.0, L - L_budget)^2
        # return L - λ*D + pen, L, D
        
        # fixed Nov28 OBJECTIVE
        gap = L - L_budget
        return gap^2 - λ*D, L, D
    end

    return (; objective, p_var0, κ_grid, kappa_tilde_on_grid, L_star, L_budget)
end

function run_mdcurve_for_seed(seed::Int; outroot::String=OUTROOT, delta_frac::Float64=0.01)
    run_dir = joinpath(outroot, "run_$(seed)")
    @assert isdir(run_dir) "No directory $run_dir. Train seed $seed first."
    built = build_run(seed)  # rebuild nnκ/states consistently
    md = build_mdcurve(seed, run_dir, built; delta_frac=delta_frac)

    ad = Optimization.AutoZygote()
    f_only = (x, _)->(md.objective(x; λ=1.0, μ=200.0)[1])
    optf   = Optimization.OptimizationFunction(f_only, ad)
    optprob= Optimization.OptimizationProblem(optf, getdata(md.p_var0))

    println("[MDC] Starting ADAM optimization...")
    res1 = Optimization.solve(optprob, ADAM(1e-3); maxiters=5_000)
    println("[MDC] Starting L-BFGS optimization...")
    res2 = Optimization.solve(Optimization.OptimizationProblem(optf, res1.u),
                              LBFGS(linesearch=BackTracking()); maxiters=2_000)
    println("[MDC] Optimization finished.")

    Jfin, Lfin, Dfin = md.objective(res2.u; λ=1.0, μ=200.0)

    # Δloss% now computed vs baseline MSE (consistent with budget)
    dloss_pct_mse = (Lfin - md.L_star_mse) / md.L_star_mse * 100.0

    # Save CSV summary (note: include both training loss and MSE baseline for clarity)
    CSV.write(joinpath(run_dir, "mdcurve_summary.csv"),
        DataFrame(; seed=seed,
                   L_star_train = md.L_star,      # training loss incl. regularizer (for reference)
                   L_star_mse   = md.L_star_mse,  # baseline data-fit used for budget & Δloss%
                   L_budget     = md.L_budget,
                   L_alt        = Lfin,
                   rel_dloss_percent = dloss_pct_mse,
                   norm_curve_distance = Dfin))

    # Overlay plot: κ baseline vs κ̃ along training times
    # fix wrong axes object (was md.p_img_0.p0_var0)
    κ̃_grid = md.kappa_tilde_on_grid(ComponentArray(res2.u, getaxes(md.p_var0)))
    plt = plot(tdata, md.κ_grid, label="κ baseline", xlabel="time", ylabel="κ",
               title="Seed $seed – minimally disruptive κ")
    plot!(plt, tdata, κ̃_grid, label="κ̃ (min-disrupt)", linestyle=:dash)
    ann = "Δloss (MSE) = $(round(dloss_pct_mse, digits=2))% | Δκ_norm = $(round(Dfin, digits=3))"
    annotate!(plt, tdata[2], maximum(md.κ_grid), text(ann, 8))
    savefig(plt, joinpath(run_dir, "mdcurve_kappa_overlay.png"))

    return (; Lfin, Dfin)
end

# Run MDC for one seed and save summary + overlay plot
function run_mdcurve_for_seed_old(seed::Int; outroot::String=OUTROOT, delta_frac::Float64=0.01)
    run_dir = joinpath(outroot, "run_$(seed)")
    @assert isdir(run_dir) "No directory $run_dir. Train seed $seed first."
    built = build_run(seed)  # to rebuild nnκ/states consistently
    md = build_mdcurve(seed, run_dir, built; delta_frac=delta_frac)

    ad = Optimization.AutoZygote()
    f_only = (x, _)->(md.objective(x; λ=1.0, μ=200.0)[1])
    optf   = Optimization.OptimizationFunction(f_only, ad)
    optprob= Optimization.OptimizationProblem(optf, getdata(md.p_var0))

    println("[MDC] Starting ADAM optimization...") # 
    res1 = Optimization.solve(optprob, ADAM(1e-3); maxiters=5_000)
    println("[MDC] Starting L-BFGS optimization...") # 
    res2 = Optimization.solve(Optimization.OptimizationProblem(optf, res1.u),
                              LBFGS(linesearch=BackTracking()); maxiters=2_000)
    println("[MD] Optimization finished.") #


    Jfin, Lfin, Dfin = md.objective(res2.u; λ=1.0, μ=200.0)

    # Save CSV summary
    println("[MDC] Saving results...") # 
    CSV.write(joinpath(run_dir, "mdcurve_summary.csv"),
        DataFrame(; seed=seed,
                   L_star = md.L_star,
                   L_budget = md.L_budget,
                   L_alt = Lfin,
                   rel_dloss_percent = (Lfin - md.L_star) / md.L_star * 100.0,
                   norm_curve_distance = Dfin))

    # Overlay plot: κ baseline vs κ̃ along training times
    κ̃_grid = md.kappa_tilde_on_grid(ComponentArray(res2.u, getaxes(md.p_img_0.p0_var0)))
    plt = plot(tdata, md.κ_grid, label="κ baseline", xlabel="time", ylabel="κ",
               title="Seed $seed – minimally disruptive κ")
    plot!(plt, tdata, κ̃_grid, label="κ̃ (min-disrupt)", linestyle=:dash)
    ann = "Δloss = $(round((Lfin - md.L_star)/md.L_star*100, digits=2))% | Δκ_norm = $(round(Dfin, digits=3))"
    annotate!(plt, tdata[2], maximum(md.κ_grid), text(ann, 8))
    savefig(plt, joinpath(run_dir, "mdcurve_kappa_overlay.png"))

    return (; Lfin, Dfin)
end

# Helper: run MDC on the best seed by lowest LBFGS loss
function run_mdcurve_on_best(; outroot::String=OUTROOT, delta_frac::Float64=0.01)
    sum_path = joinpath(outroot, "ensemble_summary.csv")
    @assert isfile(sum_path) "Missing ensemble_summary.csv in $outroot"
    df_sum = CSV.read(sum_path, DataFrame)
    best_row  = sort(df_sum, :lbfgs_loss)[1, :]
    best_seed = Int(best_row.seed)
    println("[MDC] Running minimally disruptive curve on best seed = $best_seed with δ=$(round(delta_frac*100, digits=2))%")
    run_mdcurve_for_seed(best_seed; outroot=outroot, delta_frac=delta_frac)
end


# Entry point
function main()
    println("[INFO] OUTROOT = ", OUTROOT)
    println("[INFO] Starting PEM-UDE κ ensemble ($(N_RUNS) runs) @ ", Dates.now())
    df_sum = run_ensemble(; N=N_RUNS, outroot=OUTROOT)
    println("[INFO] Done @ ", Dates.now(), " → ", joinpath(OUTROOT, "ensemble_summary.csv"))

    # Now safely generate plots (sequential)
    generate_plots(; outroot=OUTROOT, N=N_RUNS)


    # (MDC): run minimally disruptive curve on the best seed
    try
        run_mdcurve_on_best(; outroot=OUTROOT, delta_frac=0.01)  # 1% budget
    catch e
        @warn "[MDC] Skipped (run manually if desired)" error=e
    end

end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
