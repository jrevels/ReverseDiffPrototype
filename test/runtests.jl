const TESTDIR = dirname(@__FILE__)

test_println(kind, f, pad = "  ") = println(pad, "testing $(kind): `$(f)`...")

println("running TapeTests...")
t = @elapsed include(joinpath(TESTDIR, "TapeTests.jl"))
println("done (took $t seconds).")

println("running TrackedTests...")
t = @elapsed include(joinpath(TESTDIR, "TrackedTests.jl"))
println("done (took $t seconds).")

println("running MacrosTests...")
t = @elapsed include(joinpath(TESTDIR, "MacrosTests.jl"))
println("done (took $t seconds).")

println("running ChainRulesTests...")
t = @elapsed include(joinpath(TESTDIR, "ChainRulesTests.jl"))
println("done (took $t seconds).")

println("running ScalarTests...")
t = @elapsed include(joinpath(TESTDIR, "derivatives/ScalarTests.jl"))
println("done (took $t seconds).")

println("running LinAlgTests...")
t = @elapsed include(joinpath(TESTDIR, "derivatives/LinAlgTests.jl"))
println("done (took $t seconds).")

println("running ElementwiseTests...")
t = @elapsed include(joinpath(TESTDIR, "derivatives/ElementwiseTests.jl"))
println("done (took $t seconds).")

println("running ArrayFunctionTests...")
t = @elapsed include(joinpath(TESTDIR, "derivatives/ArrayFunctionTests.jl"))
println("done (took $t seconds).")

println("running GradientTests...")
t = @elapsed include(joinpath(TESTDIR, "api/GradientTests.jl"))
println("done (took $t seconds).")

println("running JacobianTests...")
t = @elapsed include(joinpath(TESTDIR, "api/JacobianTests.jl"))
println("done (took $t seconds).")

println("running HessianTests...")
t = @elapsed include(joinpath(TESTDIR, "api/HessianTests.jl"))
println("done (took $t seconds).")

println("running ConfigTests...")
t = @elapsed include(joinpath(TESTDIR, "api/ConfigTests.jl"))
println("done (took $t seconds).")

println("running CompatTests...")
t = @elapsed include(joinpath(TESTDIR, "compat/CompatTests.jl"))
println("done (took $t seconds).")
