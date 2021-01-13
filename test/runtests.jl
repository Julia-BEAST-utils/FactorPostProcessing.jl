using Test, SafeTestsets

@safetestset "FactorPostProcessing.jl test" begin include("svd_tests.jl") end
