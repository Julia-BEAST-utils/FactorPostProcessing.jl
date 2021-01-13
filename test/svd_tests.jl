using FactorPostProcessing, Test, Random
using BeastUtils.Logs, LinearAlgebra

Random.seed!(666)

const TOL = 1e-10
const POS_COLS = [9, 5, 13]

function absmax(x::AbstractArray{T}, y::AbstractArray{T}) where T <: Real
    return maximum(abs.(x - y))
end

function svd_process(L_data::Matrix{Float64}, F_data::Matrix{Float64}, row::Int, k::Int, p::Int, n::Int)
    L_vec = @view L_data[row, :]
    L = Matrix(reshape(L_vec, p, k))
    Lsvd = svd(L)
    F_vec = @view F_data[row, :]
    F = reshape(F_vec, k, n)
    F_rotated = Lsvd.Vt * F
    S = Lsvd.S
    Vt = Lsvd.Vt
    U = Lsvd.U
    F_rotated = Vt * F
    return (sv = Lsvd.S, V = Lsvd.U, F_rotated)
end



states = 1000
K = 3
P = 20
N = 100

L_data = randn(states, K * P)
F_data = randn(states, K * N)

L_labels = vec(["L$j$i" for i = 1:K, j=1:P])
F_labels = vec(["factors.taxon_$i.$j" for j=1:K, i = 1:N])

cols = [["posterior", "prior", "likelihood"];
        ["factorPrecision$i" for i = 1:P]; L_labels; F_labels]
data = [randn(states, 3 + P) L_data F_data]

log_path = joinpath(@__DIR__, "testFactorPostProcessing.log")
svd_path = joinpath(@__DIR__, "testFactorPostProcessingSVD.log")

@assert !isfile(log_path)
@assert !isfile(svd_path)

Logs.make_log(log_path, data, cols)
svd_logs(log_path, svd_path, rotate_factors=true)

svd_cols, svd_data = get_log(svd_path, burnin=0.0)

rm(log_path)
rm(svd_path)

L_cols = findall(startswith("L"), svd_cols)
V_cols = findall(startswith("V"), svd_cols)
sv_cols = findall(startswith("sv"), svd_cols)
F_cols = findall(startswith("factors"), svd_cols)

for state = 1:states
    L_actual = reshape(svd_data[state, L_cols], P, K)
    V_actual = reshape(svd_data[state, V_cols], P, K)
    sv_actual = svd_data[state, sv_cols]
    F_actual = reshape(svd_data[state, F_cols], K, N)
    S_expected, V_expected, F_expected = svd_process(L_data, F_data, state, K, P, N)
    L_expected = V_expected * Diagonal(S_expected)
    @test absmax(sv_actual, S_expected) < TOL

    signs = ones(K)
    for k = 1:K
        if L_expected[POS_COLS[k], k] < 0.0
            signs[k] = -1.0
        end
    end

    D = Diagonal(signs)
    L_expected = L_expected * D
    V_expected = V_expected * D
    F_expected = D * F_expected

    @test absmax(V_actual, V_expected) < TOL
    @test absmax(L_actual, L_expected) < TOL
    @test absmax(F_actual, F_expected) < TOL
end

