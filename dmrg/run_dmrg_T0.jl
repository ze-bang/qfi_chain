#!/usr/bin/env julia
# run_dmrg_T0.jl --- Ground-state DMRG for J1-J2-J3 + observable JSON dump.
#
# Usage:
#   julia --project=. run_dmrg_T0.jl --phase U --N 32 --chi-max 400 --out out.json
#   julia --project=. run_dmrg_T0.jl --J1 1 --J2 0.5 --J3 0 --label DC_t050 --N 32 --out out.json
#
# Output JSON contains:
#   metadata      : phase, N, χ_max, sweeps, ε_trunc
#   convergence   : E0, |dE|, max truncation error, final maxlinkdim
#   observables   : output of `compute_observables(psi)`
# Also writes an MPS checkpoint next to the JSON as `*_psi.h5`.

using ArgParse
using HDF5
using ITensors, ITensorMPS
using JSON3
using Printf
using Random

include(joinpath(@__DIR__, "hamiltonian.jl"))
include(joinpath(@__DIR__, "observables.jl"))

function parse_cli()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--phase";    arg_type = String;  default = ""; help = "D|C|U reference point"
        "--J1";       arg_type = Float64; default = NaN; help = "custom period-4 J1 bond coupling"
        "--J2";       arg_type = Float64; default = NaN; help = "custom period-4 J2 bond coupling"
        "--J3";       arg_type = Float64; default = NaN; help = "custom period-4 J3 bond coupling"
        "--label";    arg_type = String;  default = ""; help = "metadata label for custom coupling point"
        "--N";        arg_type = Int;     required = true
        "--chi-max";  arg_type = Int;     default = 400
        "--sweeps";   arg_type = Int;     default = 30
        "--tol";      arg_type = Float64; default = 1e-10
        "--r-max";    arg_type = Int;     default = 3
        "--seed";     arg_type = Int;     default = 1234
        "--out";      arg_type = String;  required = true
    end
    return parse_args(s)
end

function main()
    args = parse_cli()
    Random.seed!(args["seed"])

    phase   = args["phase"]
    label   = isempty(args["label"]) ? phase : args["label"]
    N       = args["N"]
    chi_max = args["chi-max"]
    nsweep  = args["sweeps"]
    tol     = args["tol"]
    r_max   = args["r-max"]
    out     = args["out"]

    use_phase = !isempty(phase)
    use_custom = !(isnan(args["J1"]) || isnan(args["J2"]) || isnan(args["J3"]))
    use_phase ⊻ use_custom || error("provide exactly one of --phase D|C|U or custom --J1 --J2 --J3")

    J1, J2, J3 = use_phase ? PHASE_COUPLINGS[phase] : (args["J1"], args["J2"], args["J3"])

    @info "Building J1J2J3 chain" phase label J1 J2 J3 N chi_max nsweep
    sites = j1j2j3_sites(N; conserve_sz = true)
    H     = j1j2j3_mpo(sites, J1, J2, J3)

    # Initial Néel state in S^z_tot=0 sector.
    state = [isodd(i) ? "Up" : "Dn" for i in 1:N]
    psi0  = MPS(sites, state)

    # DMRG schedule: ramp χ.
    sweeps = Sweeps(nsweep)
    setmaxdim!(sweeps,
        min(chi_max, 50),  min(chi_max, 100),
        min(chi_max, 200), min(chi_max, 400), chi_max)
    setcutoff!(sweeps, tol)
    setnoise!(sweeps, 1e-7, 1e-9, 0.0)

    @info "Running DMRG"
    energy, psi = dmrg(H, psi0, sweeps; outputlevel = 1)

    final_chi = maxlinkdim(psi)
    trunc     = tol  # ITensor returns per-bond cutoff; we report the cap
    @info "DMRG done" energy final_chi

    @info "Computing observables"
    obs = compute_observables(psi; r_max = r_max)

    payload = Dict(
        "metadata" => Dict(
            "kind"     => "T0_DMRG",
            "phase"    => phase,
            "label"    => label,
            "J1"       => J1,
            "J2"       => J2,
            "J3"       => J3,
            "N"        => N,
            "chi_max"  => chi_max,
            "sweeps"   => nsweep,
            "tol"      => tol,
            "seed"     => args["seed"],
        ),
        "convergence" => Dict(
            "E0"          => energy,
            "final_chi"   => final_chi,
            "trunc_cap"   => trunc,
        ),
        "observables" => obs,
    )

    mkpath(dirname(out))
    open(out, "w") do io
        JSON3.write(io, payload)
    end
    @info "Wrote" out

    # Save the converged state so F_EP / eta post-processing can rebuild
    # symmetry-adapted operator bases without repeating DMRG.
    psi_path = replace(out, r"\.json$" => "_psi.h5")
    h5open(psi_path, "w") do f
        write(f, "psi", psi)
    end
    @info "Wrote MPS checkpoint" psi_path
end

isinteractive() || main()
