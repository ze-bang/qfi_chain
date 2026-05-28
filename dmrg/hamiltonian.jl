"""
hamiltonian.jl --- period-4 J1-J2-J3 spin-1/2 chain MPO builder.

The model is a nearest-neighbor Heisenberg chain with a repeating
period-4 bond pattern

    J1, J2, J1, J3, J1, J2, J1, J3, ...

so J1/J2/J3 label bond classes, not first-/second-/third-neighbor
couplings. Open boundary conditions are used throughout.

Reference points:
    (D) Dimer / isolated singlets    (J1,J2,J3) = (1.0, 0.0, 0.0)
    (C) Four-site clusters           (J1,J2,J3) = (1.0, 1.0, 0.0)
    (U) Uniform Heisenberg chain     (J1,J2,J3) = (1.0, 1.0, 1.0)
"""

using ITensors, ITensorMPS

const PHASE_COUPLINGS = Dict(
    "D" => (1.0, 0.0, 0.0),
    "C" => (1.0, 1.0, 0.0),
    "U" => (1.0, 1.0, 1.0),
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

Build the MPO for the period-4 nearest-neighbor bond pattern:
bond i=(i,i+1) has coupling J1 for i mod 4 in {1,3}, J2 for i mod 4 = 2,
and J3 for i mod 4 = 0 (Julia 1-indexed sites).
"""
function j1j2j3_mpo(sites, J1::Real, J2::Real, J3::Real)
    N = length(sites)
    os = OpSum()
    for j in 1:(N - 1)
        Jr = if mod(j, 4) in (1, 3)
            J1
        elseif mod(j, 4) == 2
            J2
        else
            J3
        end
        Jr == 0 && continue
        os += Jr/2, "S+", j, "S-", j+1
        os += Jr/2, "S-", j, "S+", j+1
        os += Jr,   "Sz", j, "Sz", j+1
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
