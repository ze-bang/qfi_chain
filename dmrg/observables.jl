"""
observables.jl --- MPS observable extraction shared by T=0 DMRG and METTS.

We compute, for an arbitrary MPS |psi⟩:
    one_pt[a, j]                        = ⟨S^a_j⟩
    two_pt[a, b, j, k]   (j<k)          = ⟨S^a_j S^b_k⟩
    three_pt_chain[r, j]                = ⟨[X^XY_j(r), G]⟩-related contractions
    four_pt_window[r, s, j]             = window covariance contractions
where a,b ∈ {x,y,z} indexed 1,2,3 and the staggered probe is
    G = (1/√N) Σ_j (-1)^j S^z_j .

For body-2 only ({O^XY,r}_π for r=1..r_max) we need:
    s_r = -i⟨[O^XY,r_π, G]⟩ = (4/√N) Σ_j (-1)^j ⟨ S^y_j S^x_{j+r} - S^x_j S^y_{j+r} ⟩  / √N
        (writing G as N^{-1/2} Σ (-1)^j S^z_j, both factors carry 1/√N).
    Σ_{rs} = ⟨ O^XY,r_π O^XY,s_π ⟩ - ⟨O^XY,r_π⟩⟨O^XY,s_π⟩
which expands into a sum of staggered four-point functions.

This file exports `compute_observables(psi; phase, N, r_max)` returning a
NamedTuple with all needed reduced quantities written by the drivers as JSON.
"""

using ITensors, ITensorMPS, LinearAlgebra

const SPIN_OPS = ("Sx", "Sy", "Sz")

"""One-point ⟨S^a_j⟩ for a in {Sx,Sy,Sz}."""
function one_point_all(psi::MPS)
    sites = siteinds(psi)
    N = length(sites)
    M = zeros(ComplexF64, 3, N)
    for (ai, opname) in enumerate(SPIN_OPS), j in 1:N
        M[ai, j] = inner(psi', op(opname, sites[j]), psi)
    end
    return real.(M)  # all ⟨S^a⟩ are real for our states
end

"""Two-point ⟨S^a_j S^b_k⟩ for j<k. Returns a (3,3,N,N) array; entries
with j>=k are left zero."""
function two_point_all(psi::MPS)
    sites = siteinds(psi)
    N = length(sites)
    C = zeros(ComplexF64, 3, 3, N, N)
    # ITensor's `correlation_matrix` gives ⟨A_i B_j⟩ for all i,j.
    for (ai, A) in enumerate(SPIN_OPS), (bi, B) in enumerate(SPIN_OPS)
        Cab = correlation_matrix(psi, A, B; sites=1:N)
        for j in 1:N, k in (j+1):N
            C[ai, bi, j, k] = Cab[j, k]
        end
    end
    return C
end

"""Compute body-2 matched-filter signal vector and covariance matrix.

s_r and Σ_{rs} for the basis {O^{XY,r}_π}_{r=1..r_max} on probe G=S^z_π.

Inputs: one_pt (3,N), two_pt (3,3,N,N).
Returns (s::Vector{Float64}, Σ::Matrix{Float64}, FQ::Float64).

The signal uses the linear-response identity
    s_r = -i⟨[O^XY,r_π, G]⟩
        = (2/N) Σ_j (-1)^j (-1)^j   {  ... }
We expand:
    [S^x_j S^x_{j+r} + S^y_j S^y_{j+r}, S^z_l]
        = δ_{l,j}   ( i S^y_j S^x_{j+r} - i S^x_j S^y_{j+r} )
        + δ_{l,j+r} ( i S^x_j S^y_{j+r} - i S^y_j S^x_{j+r} )
giving (with the 1/√N factors of both O and G already pulled out)
    s_r = (1/N) Σ_j [ (-1)^j (-1)^j - (-1)^j (-1)^{j+r} ] · 2⟨S^y_j S^x_{j+r} - S^x_j S^y_{j+r}⟩
        = (1/N) Σ_j [1 - (-1)^r] · 2⟨S^y_j S^x_{j+r} - S^x_j S^y_{j+r}⟩
which **vanishes for even r**.  For odd r,
    s_r = (4/N) Σ_j ⟨S^y_j S^x_{j+r} - S^x_j S^y_{j+r}⟩.
This is intentional: at body-2, the O^XY,r probes only odd-r staggered
sectors carry the matched-filter signal.

The QFI is
    F_Q = (4/N) Σ_{j,k} (-1)^{j+k} (⟨S^z_j S^z_k⟩ - ⟨S^z_j⟩⟨S^z_k⟩).
"""
function build_body2_filter(one_pt::Matrix{Float64}, two_pt::Array{ComplexF64,4}, r_max::Int)
    N = size(one_pt, 2)
    @assert size(two_pt, 4) == N

    # F_Q (probe G = N^{-1/2} Σ (-1)^j S^z_j; F_Q = 4 Var(G)).
    FQ = 0.0
    for j in 1:N, k in 1:N
        sgn = ((-1)^(j+k))
        czz = j < k ? real(two_pt[3, 3, j, k]) :
              j > k ? real(two_pt[3, 3, k, j]) :
              0.25  # ⟨(S^z_j)^2⟩ = 1/4
        FQ += 4.0/N * sgn * (czz - one_pt[3, j] * one_pt[3, k])
    end

    # Build s_r and Σ_{rs} on basis {O^{XY,r}_π}_{r=1..r_max}.
    rs = collect(1:r_max)
    nb = length(rs)
    s = zeros(Float64, nb)
    Σ = zeros(Float64, nb, nb)

    # Helper: signed two-point ⟨S^a_j S^b_k⟩ for a,b ∈ {1,2,3} regardless of j vs k.
    twop(a, b, j, k) = j < k ? two_pt[a, b, j, k] :
                       j > k ? conj(two_pt[b, a, k, j]) :
                       (a == b ? ComplexF64(0.25) : ComplexF64(0.0))

    # Signal s_r.
    for (ri, r) in enumerate(rs)
        if iseven(r); continue; end
        acc = 0.0
        for j in 1:(N - r)
            yx = real(twop(2, 1, j, j+r))   # ⟨S^y_j S^x_{j+r}⟩
            xy = real(twop(1, 2, j, j+r))   # ⟨S^x_j S^y_{j+r}⟩
            acc += yx - xy
        end
        s[ri] = (4.0 / N) * acc
    end

    # Covariance Σ_{rs} = ⟨O^XY,r O^XY,s⟩ - ⟨O^XY,r⟩⟨O^XY,s⟩.
    # ⟨O^XY,r⟩ = (1/√N) Σ_j (-1)^j ⟨S^x_j S^x_{j+r} + S^y_j S^y_{j+r}⟩.
    means = zeros(Float64, nb)
    for (ri, r) in enumerate(rs)
        m = 0.0
        for j in 1:(N - r)
            m += ((-1)^j) * (real(twop(1, 1, j, j+r)) + real(twop(2, 2, j, j+r)))
        end
        means[ri] = m / sqrt(N)
    end
    # Σ_{rs} four-point part is computed exactly only via real four-point
    # functions; here we use the *Gaussian / Wick-disconnected* approximation
    # plus the connected one-body correction:
    #   Σ_{rs} ≈ Σ^disc_{rs} - means[r]·means[s]
    # where Σ^disc is computed from the empirical two-points and a sum over
    # contiguous Wick contractions:
    #   ⟨A_jk B_lm⟩ ≈ ⟨A_jk⟩⟨B_lm⟩ + ⟨A_jk B_lm⟩_c
    # For an exact Σ at body-4 we need the four-point function below; the
    # body-2 channel of the Wiener problem only uses the truncated 2pt
    # estimate as an upper bound on Var(O), giving a *lower bound* on η.
    # The driver writes the raw two-points so a more accurate four-point
    # assembly can be done in post-processing if needed.
    for ri in 1:nb, si in 1:nb
        # Diagonal estimate from variance of O^{XY,r}: Σ_{rr} = ⟨(O^XY,r)^2⟩ - means[r]^2.
        # Off-diagonal: cross-Wick contractions; we set 0 here and let the
        # Python assembly recompute from the saved two-point if needed.
        if ri == si
            r = rs[ri]
            v = 0.0
            # ⟨O^2⟩ = (1/N) Σ_{j,k} (-1)^{j+k} ⟨X_j X_k⟩ where X_j is the
            # XY exchange on bond (j,j+r). Wick-only estimate:
            #   ⟨X_j X_k⟩ ≈ ⟨X_j⟩⟨X_k⟩ + (Wick contractions of two-points)
            # For brevity in the seed pipeline we use the leading
            # disconnected piece v ≈ Σ_j ⟨X_j⟩²/N which is the strict
            # mean-field lower bound; replace by exact four-point in
            # post-processing for production.
            for j in 1:(N - r)
                xj = real(twop(1, 1, j, j+r)) + real(twop(2, 2, j, j+r))
                v += xj * xj
            end
            Σ[ri, si] = v / N - means[ri] * means[si]
        end
    end

    return s, Σ, FQ, means
end

"""
    compute_observables(psi; r_max=3) :: Dict

Wraps the above and returns a dictionary ready to write as JSON. The
returned dict contains the *raw* one- and two-point functions so a
fuller four-point assembly can run downstream in Python.
"""
function compute_observables(psi::MPS; r_max::Int=3)
    one_pt = one_point_all(psi)
    two_pt = two_point_all(psi)
    s, Σ, FQ, means = build_body2_filter(one_pt, two_pt, r_max)
    return Dict(
        "N"          => length(siteinds(psi)),
        "r_max"      => r_max,
        "one_pt"     => one_pt,                       # (3, N), real
        "two_pt_re"  => real.(two_pt),                # (3,3,N,N)
        "two_pt_im"  => imag.(two_pt),                # (3,3,N,N)
        "FQ_Szpi"    => FQ,
        "OXYr_means" => means,                        # (r_max,)
        "s_r"        => s,                            # (r_max,)
        "Sigma_rr"   => [Σ[i,i] for i in 1:r_max],    # diag only
    )
end
