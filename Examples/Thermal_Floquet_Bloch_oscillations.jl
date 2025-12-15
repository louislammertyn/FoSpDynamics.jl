using Revise
using FoSpDynamics
using Plots
using Interpolations
using LinearAlgebra
using OrdinaryDiffEq

function compare_energies(ω::Float64, T::Float64)
    ħ = 1.054571817e-34   # Js
    kB = 1.380649e-23     # J/K
    return (ħ * ω) / (kB * T)
end

## set constants ##
begin
ϕ = π/2
ω = 2*π * 440
Δ = 50
ω_d = ω + Δ
κ =  Δ

T = 2*π / ω
t_i, t_e= 0., 1.
dt = 1e-3
ts = t_i:dt:t_e


E_ratio = compare_energies(ω, 100e-9)
Temp = ω / E_ratio
β = 1/ (Temp)
L = 10
end;
β
## define time dependency and initialise the interpolation functions for the time evolution ##
begin
f_t(ts) = κ .* cos.(ω_d .*ts.+ϕ) ;
triv(ts) = ones(ComplexF64,length(ts));
interp_f_t = linear_interpolation(ts, f_t(ts));
interp_trivial(t) = 1.;

## set the times at which to save the simulation steps 
strob_ts = tuple(collect(t_i:(2π/ω_d): t_e)...);
save_ts = tuple(collect(t_i:(2π/ω_d)/(2): t_e)...);
end;


begin
## initialisation of lattice and hilbert space ##
geometry = (L,)
V = U1FockSpace(geometry, 1,1)
states = all_states_U1_O(V)
lattice = Lattice(geometry)

## Define the conditions for the hopping structure of the model ##
function hop_int(s1::Int, s2::Int)
    diff = abs(s1-s2)
    return diff==0 ? 1 : sin(diff*π/2)/(diff*π/2)
end



#### defining conditions ####
function fhop_2body(sites_tuple)
    s1, s2 = sites_tuple
    J = hop_int(s1[1],s2[1])
    return J
end

function fonsite_2body(sites_tuple)
    s1, s2 = sites_tuple
    o =  1 : 0
    return s1==s2 ? ((s1[1]-1)  * ω ) : zero(ComplexF64)
end


condition1 = (fhop_2body, )
condition2 = (fonsite_2body,)

## make the tensors and with them the operators that form the Hamiltonian ##
t1 = ManyBodyTensor_init(ComplexF64, V, 1, 1)
tens_hop = fill_nbody_tensor(t1, lattice, condition1)

t2 = ManyBodyTensor_init(ComplexF64, V, 1, 1)
tens_onsite = fill_nbody_tensor(t2, lattice, condition2)


Hop = nbody_Op(V, lattice, tens_hop)
H_onsite = nbody_Op(V, lattice, tens_onsite)
end;

begin
Hop_m = calculate_matrix_elements_parallel(states, Hop)
H_onsite_m = calculate_matrix_elements_parallel(states, H_onsite)

H_0 = Hop_m + H_onsite_m

## Choose an initial state (thermal state) ##
ρ = thermal_ρ_matrix(Hop + H_onsite, β, lattice)
end;

thermal_ρ_matrix(Hop + H_onsite, β, lattice)


## Time evolution where the lists indicate the time dependent functions and their corresponding operators ## 
begin
interps = [interp_trivial, interp_f_t]
ops = [H_onsite_m, Hop_m]

sol = Time_Evolve_thermal_ρ_TD(ρ, (ops, interps), (t_i,t_e), save_ts; rtol = 1e-9, atol = 1e-9, solver = Vern7())
end;

## Plot the solution ##
begin
pl = plot(
    xlabel = "t",           # replace "units" with physical units if any
    ylabel = "Center of Mass <λ> ", 
    title = "center of mass motion",
    legend = false,
    grid = true,
    framestyle = :box
);
for s in eachindex(sol.t)[1:end]
    state = reshape(sol.u[s], (L, L))
    l_com = real(tr(state * Diagonal(0:(L-1))))
    
    x = x_matrix(L)
    x_com = real(tr(state * x))

    #scatter!(pl, [sol.t[s]], [l_com], marker=:o, color=:blue, markersize=1)
    scatter!(pl,  [sol.t[s]], [x_com], marker=:o, color=:red, markersize=1)
end;
display(pl)
end;

function x_matrix(N::Int)
    x = zeros(Float64, N, N)
    for n in 1:N-1
        val = sqrt(n / 2)
        x[n, n+1] = val
        x[n+1, n] = val
    end
    return x
end
