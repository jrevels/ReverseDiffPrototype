module ChainRulesTest

using ChainRules
using DiffResults
using ReverseDiff
using Test

f(x) = sum(4x .+ 1)

function ChainRules.rrule(::typeof(f), x)
    r = f(x)
    function back(d)
        return ChainRules.NoTangent(), fill(3 * d, size(x))
    end
    return r, back
end

ReverseDiff.@grad_from_cr f

begin # test ChainRules.rrule
    input = rand(3, 3)
    output, back = ChainRules.rrule(f, input);
    _, d = back(1)
    @test output == f(input)
    @test d == fill(3, size(input))
end

begin # test ReverseDiff
    const f_tape = ReverseDiff.GradientTape(f, (rand(3, 3),))
    inputs = (rand(3, 3), )
    results = (similar(inputs[1]),)
    # all_results = map(DiffResults.GradientResult, results)
    cfg = ReverseDiff.GradientConfig(inputs)

    ReverseDiff.gradient!(results, f_tape, inputs)

    @test results[1] == fill(3, size(inputs[1]))
end

end
