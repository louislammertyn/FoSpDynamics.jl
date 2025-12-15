using Revise
using QuantumFockDynamics, LinearAlgebra

L = 15
geometry = (L,)
V = U1FockSpace(geometry, 6, 6)
lattice = Lattice(geometry)
basis = all_states_U1_O(V)

K, U = Bose_Hubbard_H(V, lattice)
@time H_m = calculate_matrix_elements(K + U, basis)
basis