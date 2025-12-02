

function thermal_ρ_matrix(H::MultipleFockOperator, β::Float64, lattice::Lattice)
    mb_tnsrs = extract_nbody_tensors(H, lattice)
    @assert (length(mb_tnsrs)==1) && (mb_tnsrs[1].domain + mb_tnsrs[1].codomain == 2) "Thermal state is only explicitely defined for single particle Hamiltonians"
    
    Ham = vectorize_tensor(mb_tnsrs[1], lattice)
    @assert Ham == Ham' "H is not Hermitian"

    es, vs = eigen(Hermitian(Ham))
    exp_es = (es .* -1 * β)  .|>exp |> Diagonal
    

    Z = tr(exp_es)
    ρ = vs * exp_es * vs' / Z
    
    @assert isapprox(ρ, ρ';atol=1e-9)
    @assert all(real.(eigvals(ρ)) .>= -1e-12)
    @assert isapprox(tr(ρ), 1.; atol=1e-9)
    return ρ
end

function thermal_exp(ρ::Matrix{ComplexF64}, Op::Matrix{ComplexF64})
    return tr(ρ * Op)
end

function Liouvillian_Super(Op::Matrix{ComplexF64})
    I_n = Matrix{ComplexF64}(I, size(Op))
    return  (Op ⊗ I_n) - (I_n ⊗ Transpose(Op))
end


function Time_Evolution_thermal_ρ_Liouv(init_ρ::Matrix{ComplexF64}, H::AbstractMatrix{ComplexF64},
                        tspan::Tuple{Float64, Float64};
                        rtol::Float64 = 1e-9, atol::Float64 = 1e-9,
                        solver = Vern7())
    L = Liouvillian_Super(H)
    sol = Time_Evolution(vec(init_ρ), H, tspan; rtol=rtol, atol=atol, solver=solver)
    return sol
end

function Time_Evolution_thermal_ρ_TD_Liouv(init_ρ::Matrix{ComplexF64},
                           ops::NTuple{N, Matrix{ComplexF64}}, f_ts::Tuple{Vararg{<:Function, N}},
                           tspan::Tuple{Float64, Float64}, tpoints::NTuple{M, Float64};
                           rtol::Float64 = 1e-9, atol::Float64 = 1e-9,
                           solver = Vern7()) where {N, M}
    ρ_v = vec(init_ρ)

    Liouvillians =Vector{Matrix{ComplexF64}}()
    
    
    for (O, f) in zip(ops, f_ts)
        push!(Liouvillians, Liouvillian_Super(O))
        push!(interps, f)
    end
    Liouvillians_and_interps = (Liouvillians, interps)
    sol = Time_Evolution_TD(ρ_v,
                           Tuple(Liouvillians), f_ts,
                           tspan, tpoints;
                           rtol=rtol, atol=atol,
                           solver=solver)
    return sol 
end


function Time_Evolution_thermal_ρ_TD_VN(init_ρ::Matrix{ComplexF64},
                           ops::NTuple{N, Matrix{ComplexF64}}, f_ts::Tuple{Vararg{<:Function, N}},
                           tspan::Tuple{Float64, Float64}, tpoints::NTuple{M, Float64};
                           rtol::Float64 = 1e-9, atol::Float64 = 1e-9,
                           solver = Vern7()) where {N, M}
    
    sol = Time_Evolution_TD_VN(init_ρ,
                           ops, f_ts,
                           tspan, tpoints;
                           rtol=rtol, atol=atol,
                           solver=solver)
    return sol
end

function Unitary_Ev_ρ_TD(init_ρ::Matrix{ComplexF64}, ops::NTuple{N, Matrix{ComplexF64}}, f_ts::Tuple{Vararg{<:Function, N}}, ti::Float64, te::Float64, dt::Float64) where {N}
    U = Unitary_Ev_TD(ops, f_ts, ti, te, dt)
    return U * init_ρ * U'
end

function Unitary_Ev_ρ(init_ρ::Matrix{ComplexF64}, H::Matrix{ComplexF64}, ti::Float64, te::Float64)
    U = Unitary_Ev(H, ti, te)
    return U * init_ρ * U'
end
