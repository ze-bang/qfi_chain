#!/usr/bin/env julia
# run_dmrg_T0.jl --- Ground-state DMRG for J1-J2-J3 + observable JSON dump.
#
# Usage:
#   julia --project=. run_dmrg_T0.jl --phase U --N 32 --chi-max 400 --out out.json
#
# Output JSON contains:
#   metadata      : phase, N, χ_max, sweeps, ε_trunc
#   convergence   : E0, |dE|, max truncation error, final maxlinkdim
#   observables   : output of `compute_observables(psi)`

using ArgParse
using ITensors, ITensorMPS
using JSON3
using Printf
using Random

include(joinpath(@__DIR__, "hamiltonian.jl"))
include(joinpath(@__DIR__, "observables.jl"))

function parse_cli()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--phase";    arg_type = String;  required = true; help = "D|C|U"
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
    N       = args["N"]
    chi_max = args["chi-max"]
    nsweep  = args["sweeps"]
    tol     = args["tol"]
    r_max   = args["r-max"]
    out     = args["out"]

    @info "Building J1J2J3 chain" phase N chi_max nsweep
    sites = j1j2j3_sites(N; conserve_sz = true)
    H     = j1j2j3_mpo(sites, phase)

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
end

isinteractive() || main()
