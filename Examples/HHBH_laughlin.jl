using LinearAlgebra, Plots, ProgressMeter
using Revise
using QuantumFockDynamics

# Model structure #
geometry = (8, 8)
m = 4
modesLLL = prod(geometry) / m
N = (1/2) * modesLLL
cutoff = 2
V = U1FockSpace(geometry, cutoff, Int(N))
lattice = Lattice(geometry; periodic=(true, true))

Vsb = U1FockSpace(geometry, 1, 1)
basis_sp = all_states_U1_O(Vsb)
lattice.sites_v

ϕ = (1/m)*2*π
Jx = 1.
Jy = 1.
g = 4.

# ManyBodyTensor filling conditions #
function ϕ_hop_x(s_tuple)
    s1, s2 = s_tuple
     
    if periodic_neighbour(s1, s2, 1, lattice, geometry) 
        return  exp(1im * ϕ * s1[2]) * Jx
    elseif periodic_neighbour(s2, s1, 1, lattice, geometry) 
        return conj(exp(1im * ϕ * s1[2]) * Jx)
    else 
        return zero(ComplexF64)
    end
end

function ϕ_hop_y(s_tuple)
    s1, s2 = s_tuple
    if periodic_neighbour(s1, s2, 2, lattice, geometry) 
        return Jy
    elseif periodic_neighbour(s2, s1, 2, lattice, geometry) 
        return conj(Jy)
    else 
        return zero(ComplexF64)
    end
end

function int(s_tuple)
    s1, s2, s3, s4 = s_tuple
    return g * (s1==s2==s3==s4)
end

# Construct Hamiltonian #
fconditions1 = (ϕ_hop_x, ϕ_hop_y)

Thop = ManyBodyTensor_init(ComplexF64, V, 1, 1)
Thop = fill_nbody_tensor(Thop, lattice, fconditions1)

fconditions2 = (int, )
Tint = ManyBodyTensor_init(ComplexF64, V, 2, 2)
Tint = fill_nbody_tensor(Tint, lattice, fconditions2)

K = nbody_Op(V, lattice, Thop)
U = nbody_Op(V, lattice, Tint)

# Diagonalisation #

K_m = calculate_matrix_elements_parallel(basis_sp, K)
es, vs = eigen(Hermitian(K_m))
LLL = Matrix(Transpose(vs[:, 1:16]))
H_p = transform(K + U, lattice, LLL)

projected_space = U1FockSpace((16,), 1, 8)
projected_basis = all_states_U1(projected_space)
length(H_p.terms)

coeffs = []
for term in H_p.terms
    if length(term.product) == 4
        push!(coeffs, norm(term.coefficient))
    end
end
plot(sort(coeffs))
m = maximum(coeffs) 

function term_condition(Op::FockOperator)
    return (length(Op.product)==4 && norm(Op.coefficient) > m * 0.02)
end

H_p_r = reduce_terms(H_p, term_condition)
length(H_p_r.terms)

H_p_r_m = calculate_matrix_elements_parallel_sparse(projected_basis, H_p_r)

