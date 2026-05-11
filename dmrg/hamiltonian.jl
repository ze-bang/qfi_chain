"""
hamiltonian.jl --- J1-J2-J3 spin-1/2 chain MPO builder.

Three reference points used in the manuscript:
    (D) Majumdar-Ghosh dimer    (J1,J2,J3) = (1.0, 0.5, 0.0)
    (C) Cluster product          (J1,J2,J3) ≈ (0.241, 0.451, 0.308)
    (U) Uniform Heisenberg       (J1,J2,J3) = (1.0, 0.0, 0.0)

Open boundary conditions throughout.
"""

using ITensors, ITensorMPS

const PHASE_COUPLINGS = Dict(
    "D" => (1.0, 0.5, 0.0),
    "C" => (0.241, 0.451, 0.308),
    "U" => (1.0, 0.0, 0.0),
)

"""
    j1j2j3_sites(N; conserve_sz=true)

Construct a vector of `N` SpinHalf site indices with optional U(1)
conservation of S^z_total (default true; reduces bond dim drastically).
"""
function j1j2j3_sites(N::Int; conserve_sz::Bool=true)
    return siteinds("S=1/2", N; conserve_qns=conserve_sz)
end

"""
    j1j2j3_mpo(sites, J1, J2, J3)

Build the MPO H = J1 Σ S_i·S_{i+1} + J2 Σ S_i·S_{i+2} + J3 Σ S_i·S_{i+3}
on open boundary.
"""
function j1j2j3_mpo(sites, J1::Real, J2::Real, J3::Real)
    N = length(sites)
    os = OpSum()
    for r in 1:3
        Jr = (J1, J2, J3)[r]
        Jr == 0 && continue
        for j in 1:(N-r)
            os += Jr/2, "S+", j, "S-", j+r
            os += Jr/2, "S-", j, "S+", j+r
            os += Jr,   "Sz", j, "Sz", j+r
        end
    end
    return MPO(os, sites)
end

"""
    j1j2j3_mpo(sites, phase::AbstractString)

Convenience: lookup the phase tuple and build the MPO.
"""
function j1j2j3_mpo(sites, phase::AbstractString)
    haskey(PHASE_COUPLINGS, phase) || error("unknown phase $phase; expected D|C|U")
    J1, J2, J3 = PHASE_COUPLINGS[phase]
    return j1j2j3_mpo(sites, J1, J2, J3)
end
