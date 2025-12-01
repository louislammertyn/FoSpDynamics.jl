############################################################
# Exact diagonalization evolution for Fock operators
############################################################

"""
    Time_Evolution_ed(Ops_dict::Dict, t0::Float64, t1::Float64, δt::Float64)

Performs exact diagonalization time evolution for a Hamiltonian of the form:

    H(t) = ∑ₖ fₖ(t) * Oₖ

where each `Oₖ` is a Fock operator (e.g., `MultipleFockOperator`), and 
`fₖ(t)` is either a constant (time-independent) or a time-dependent 
coefficient provided as an `Interpolation` object.

# Arguments
- `Ops_dict::Dict`: mapping operators `Oₖ` → coefficients `fₖ`  
    - If `fₖ` is a `Number`, the operator is treated as time-independent.  
    - If `fₖ` is an `Interpolation`, it is evaluated at each time step.
- `t0::Float64`: initial time  
- `t1::Float64`: final time  
- `δt::Float64`: time step for the evolution

# Returns
- `U::AbstractMatrix{ComplexF64}`: total time-evolution operator from `t0` to `t1`
"""
## !!! this function needs to be altered !!! ##
function Time_Evolution_ed(Ops_dict::Dict, t0::Float64, t1::Float64, δt::Float64)
    times = t0:δt:t1            # discretized time points
    U = I                        # initialize evolution operator

    # Loop over all time steps
    for t in times
        H = ZeroFockOperator()   # initialize Hamiltonian as zero operator

        # Construct Hamiltonian at this time step
        for O in keys(Ops_dict)
            coeff = Ops_dict[O]
            if isa(coeff, Number)
                H += coeff * O                # time-independent term
            else
                # assume coeff is an Interpolation object
                H += ComplexF64(coeff(t)) * O  # evaluate at current time
            end
        end

        # Diagonalize Hamiltonian and compute time-step evolution
        es, vs = eigen(H)
        U_step = vs * Diagonal(exp.(-im .* es .* δt)) * vs'
        U = U_step * U  # accumulate total evolution
    end

    return U
end

# ==========================================================
# Time evolution using DifferentialEquations.jl (TI Hamiltonian)
# ==========================================================
"""
    Time_Evolution(init, H, tspan; rtol, atol, solver)

Integrates the time-independent Schrödinger equation:

Arguments:
- `init`: initial state vector
- `H`: Hamiltonian matrix
- `tspan`: tuple (t0, t1)
- `rtol`, `atol`: solver tolerances
- `solver`: ODE solver algorithm (default: Vern7)

Returns:
- `sol`: solution object from DifferentialEquations.jl
"""
function Time_Evolution(init::Vector{ComplexF64}, H::AbstractMatrix{ComplexF64},
                        tspan::Tuple{Float64, Float64};
                        rtol::Float64 = 1e-9, atol::Float64 = 1e-9,
                        solver = Vern7())
    prob = ODEProblem(schrodinger!, init, tspan, H)
    sol = solve(prob, solver; reltol=rtol, abstol=atol, save_everystep=false, saveat=tpoints)
    return sol
end


# ==========================================================
# Time-independent RHS for ODEProblem
# ==========================================================
"""
    schrodinger!(dψ, ψ, H, t)

Time-independent Schrödinger equation RHS:

    dψ/dt = -i H ψ

Arguments:
- `dψ`: derivative vector to update (output)
- `ψ`: current state vector
- `H`: Hamiltonian matrix
- `t`: current time (ignored)
"""
function schrodinger!(dψ::Vector{ComplexF64}, ψ::Vector{ComplexF64},
                      H::AbstractMatrix{ComplexF64}, t::Float64)
    dψ .= -1im * (H * ψ)
    return nothing
end

# ==========================================================
# Time evolution using DifferentialEquations.jl (TD Hamiltonian)
# ==========================================================
"""
    Time_Evolution_TD(init, ops_and_interps, tspan; rtol, atol, solver)

Integrates the time-dependent Schrödinger equation:

Arguments:
- `init`: initial state vector
- `ops_and_interps`: tuple of (operator matrices, interpolation functions)
- `tspan`: tuple (t0, t1)
- `rtol`, `atol`: solver tolerances
- `solver`: ODE solver algorithm (default: Vern7)

Returns:
- `sol`: solution object from DifferentialEquations.jl
"""
function Time_Evolution_TD(init::Vector{ComplexF64},
                           ops::Tuple{Matrix{ComplexF64}}, f_ts::Tuple{T},
                           tspan::Tuple{Float64, Float64}, tpoints::NTuple{N, Float64};
                           rtol::Float64 = 1e-9, atol::Float64 = 1e-9,
                           solver = Vern7()) where {T,N}
    prob = ODEProblem(schrodinger_TD!, init, tspan, (similar(init), ops, f_ts))
    sol = solve(prob, solver; reltol=rtol, abstol=atol, save_everystep=false, saveat=tpoints)
    return sol
end

# ==========================================================
# Time-dependent RHS for ODEProblem (DifferentialEquations.jl)
# ==========================================================
"""
    schrodinger_TD!(dψ, ψ, ops_and_interps, t)

Time-dependent Schrödinger equation RHS:

    dψ/dt = -i ∑_k fₖ(t) * O_k * ψ

Arguments:
- `dψ`: derivative vector to update (output)
- `ψ`: current state vector
- `ops_and_interps`: tuple of (operator matrices, interpolation functions)
- `t`: current time
"""
function schrodinger_TD!(dψ, ψ, (tmp, Ops, interps), t)
    fill!(dψ, 0)

    @inbounds for k in eachindex(Ops)
        fk = interps[k](t)
        mul!(tmp, Ops[k], ψ)
        @inbounds @simd for i in eachindex(dψ)
            dψ[i] -= im * fk * tmp[i]
        end
    end

    return nothing
end

function Heisenberg_eom(H::AbstractFockOperator, O::AbstractFockOperator)
    RHS = commutator(H, O)
    typeof(RHS) == FockOperator && (RHS = MultipleFockOperator([RHS], 0))
    return 1im * RHS
end

function Von_Neumann!(dψ, ψ, (tmp, Ops, f_ts), t)
    O = reshape(ψ, size(Ops[1]))
    dO = reshape(dψ, size(Ops[1]))
    fill!(dO, zero(ComplexF64)) 

    for (H, f) in zip(Ops, f_ts)
        α = f(t)
        mul!(tmp, H, O)
        dO .+= α * tmp 
        mul!(tmp, O, H)
        dO .-= α * tmp 
    end
    dO *= -1im
    return nothing
end

function Unitary_Ev(H::Matrix{ComplexF64}, ti::Float64, te::Float64)
    U = exp(-1im * H * (te-ti))
    return U  
end

function Unitary_Ev_TD(Ops::Tuple{Matrix{ComplexF64}}, f_ts::Tuple, ti::Float64, te::Float64, dt::Float64)
    U = Matrix{I, size(Ops[1])...}
    H_mid = similar(U)
    U_step = similar(U)
    tmp = similar(U)

    t = ti

    while t < te 
        fill!(H_mid, 0.0 + 0.0im)
        for (H, f) in zip(Ops, f_ts)
            H_mid .+= f(t + dt/2) * H
        end

        # compute U_step = exp(-i H_mid dt)
        U_step .= exp(-1im * H_mid * dt)

        # U = U_step * U, in-place via mul!
        mul!(tmp, U_step, U)
        U .= tmp

        t += dt
    end
end


# ==========================================================
# Generate a string representing the mean field code for integration
# ==========================================================








