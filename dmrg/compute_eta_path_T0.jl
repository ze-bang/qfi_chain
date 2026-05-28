#!/usr/bin/env julia
# compute_eta_path_T0.jl -- matched-filter eta_{<=2}, eta_{<=4} from saved MPS.
#
# This computes the same geometric object as the ED hierarchy:
#   eta_V = || P_{V|psi>} deltaG|psi> ||^2 / ||deltaG|psi>||^2
# for the real span V of Hermitian connected Pauli windows. For real ground
# states and G=S^z_pi, only strings with an odd number of Sy operators overlap
# the real-form target, so we enumerate just that symmetry sector.

using ArgParse
using HDF5
using ITensors, ITensorMPS
using JSON3
using LinearAlgebra
using Printf

const PAULI_OPS = Dict(1 => "Sx", 2 => "Sy", 3 => "Sz")
const PATH_ORDER = [
    "D_t000", "DC_t025", "DC_t050", "DC_t075", "C_t100",
    "CU_t025", "CU_t050", "CU_t075", "U_t100",
    "UD_t075", "UD_t050", "UD_t025", "D_close_t000",
]

function parse_cli()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--input-dir";  arg_type = String; default = "obs/T0_path"
        "--input-file"; arg_type = String; default = ""
        "--out";        arg_type = String; default = "obs/T0_path_eta.json"
        "--kmax";       arg_type = Int;    default = 4
        "--cutoff";     arg_type = Float64; default = 1e-12
        "--eig-rtol";   arg_type = Float64; default = 1e-10
    end
    return parse_args(s)
end

function symmetry_allowed_codes(w::Int)
    out = Vector{Vector{Int}}()
    function rec!(prefix::Vector{Int}, pos::Int)
        if pos > w
            prefix[1] == 0 && return
            prefix[end] == 0 && return
            # Real-form / time-reversal sector: only odd-Sy strings can
            # overlap the symplectic target for a real ground state.
            count(==(2), prefix) % 2 == 1 || return
            # U(1) sector: |psi> and deltaG|psi> have total Sz=0, so the
            # operator must contain a Delta Sz=0 component. In the Cartesian
            # Pauli basis this requires an even number of transverse factors
            # (Sx or Sy); otherwise its ladder-operator expansion has only
            # nonzero Sz charge and is orthogonal to the target block.
            count(c -> c == 1 || c == 2, prefix) % 2 == 0 || return
            push!(out, copy(prefix))
            return
        end
        for c in 0:3
            push!(prefix, c)
            rec!(prefix, pos + 1)
            pop!(prefix)
        end
    end
    rec!(Int[], 1)
    return out
end

function apply_pauli_string(psi::MPS, sites, start::Int, codes::Vector{Int})
    phi = copy(psi)
    for (off, code) in enumerate(codes)
        code == 0 && continue
        j = start + off - 1
        Oj = dense(op(PAULI_OPS[code], sites[j]))
        phi[j] = Oj * phi[j]
        noprime!(phi[j])
    end
    return phi
end

function staggered_Gpsi(psi::MPS, sites)
    N = length(psi)
    os = OpSum()
    for j in 1:N
        # Overall normalization of G cancels in eta. This matches the FQ JSON
        # convention up to the same constant used in observables.jl.
        os += (-1)^j, "Sz", j
    end
    G = MPO(os, sites)
    out = apply(G, psi; cutoff=1e-14)
    noprime!(out)
    return out
end

function centered(phi::MPS, psi::MPS)
    ev = inner(psi, phi)
    return phi - ev * psi
end

function eta_from_basis(phis::Vector{MPS}, target::MPS; eig_rtol::Float64=1e-10)
    M = length(phis)
    M == 0 && return 0.0, 0
    K = zeros(Float64, M, M)
    b = zeros(Float64, M)
    for a in 1:M
        # Real-form target t=[Im dG;-Re dG]. For a complex vector phi=u+iv,
        # <[u;v],t> = Im(<phi|dG>). This is the matched-filter signal up to
        # the same harmless overall constant for all operators.
        b[a] = imag(inner(phis[a], target))
        for c in 1:a
            val = real(inner(phis[a], phis[c]))
            K[a, c] = val
            K[c, a] = val
        end
    end
    K = 0.5 .* (K .+ K')
    evals, evecs = eigen(Symmetric(K))
    maxeval = maximum(abs.(evals))
    tol = max(1.0, maxeval) * eig_rtol
    keep = findall(>(tol), evals)
    isempty(keep) && return 0.0, 0
    coeff = evecs[:, keep]' * b
    num = sum(abs2.(coeff ./ sqrt.(evals[keep])))
    norm_target = real(inner(target, target))
    return num / norm_target, length(keep)
end

function compute_one(json_path::String; kmax::Int=4, eig_rtol::Float64=1e-10)
    payload = JSON3.read(read(json_path, String))
    meta = payload.metadata
    conv = payload.convergence
    obs = payload.observables
    psi_path = replace(json_path, r"\.json$" => "_psi.h5")

    f = h5open(psi_path, "r")
    psi = read(f, "psi", MPS)
    close(f)
    psi = dense(psi)
    sites = siteinds(psi)
    target = centered(staggered_Gpsi(psi, sites), psi)

    phis = MPS[]
    eta_by_k = Dict{String, Any}()
    rank_by_k = Dict{String, Any}()
    nops_by_k = Dict{String, Any}()

    for k in 1:kmax
        for start in 1:(length(psi) - k + 1)
            for codes in symmetry_allowed_codes(k)
                push!(phis, centered(apply_pauli_string(psi, sites, start, codes), psi))
            end
        end
        eta, rank = eta_from_basis(phis, target; eig_rtol=eig_rtol)
        eta_by_k[string(k)] = eta
        rank_by_k[string(k)] = rank
        nops_by_k[string(k)] = length(phis)
        @info "eta" file=basename(json_path) k eta rank nops=length(phis)
    end

    return Dict(
        "label" => String(meta.label),
        "N" => Int(meta.N),
        "J1" => Float64(meta.J1),
        "J2" => Float64(meta.J2),
        "J3" => Float64(meta.J3),
        "E0" => Float64(conv.E0),
        "chi" => Int(conv.final_chi),
        "FQ" => Float64(obs.FQ_Szpi),
        "eta" => eta_by_k,
        "rank" => rank_by_k,
        "nops" => nops_by_k,
    )
end

function main()
    args = parse_cli()
    indir = args["input-dir"]
    rows = Any[]
    files = isempty(args["input-file"]) ?
        sort(filter(p -> endswith(p, ".json"), readdir(indir; join=true))) :
        [args["input-file"]]
    for f in files
        occursin("_summary", basename(f)) && continue
        push!(rows, compute_one(f; kmax=args["kmax"], eig_rtol=args["eig-rtol"]))
    end
    order = Dict(label => i for (i, label) in enumerate(PATH_ORDER))
    sort!(rows, by = r -> (r["N"], get(order, r["label"], 999)))
    open(args["out"], "w") do io
        JSON3.write(io, Dict("rows" => rows))
    end
    @info "wrote" out=args["out"] n=length(rows)
end

isinteractive() || main()
