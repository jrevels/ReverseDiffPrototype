################
# addition (+) #
################

# in-place version (necessary for nested differentiation to work)

function plus!(out, x, y)
    for i in eachindex(out)
        out[i] = x[i] + y[i]
    end
    return out
end

@inline plus!(out::TrackedArray, x::TrackedArray, y::TrackedArray) = record_plus!(out, x, y)

for A in ARRAY_TYPES
    @eval @inline plus!(out::TrackedArray, x::TrackedArray, y::$(A)) = record_plus!(out, x, y)
    @eval @inline plus!(out::TrackedArray, x::$(A), y::TrackedArray) = record_plus!(out, x, y)
end

function record_plus!(out::TrackedArray, x, y)
    copyto!(value(out), value(x) + value(y))
    record!(tape(x, y), SpecialInstruction, +, (x, y), out)
    return out
end

# Base allocating version

@inline Base.:+(x::TrackedArray{X,D}, y::TrackedArray{Y,D}) where {X,Y,D} = record_plus(x, y, D)

for A in ARRAY_TYPES
    @eval @inline Base.:+(x::TrackedArray{V,D}, y::$(A)) where {V,D} = record_plus(x, y, D)
    @eval @inline Base.:+(x::$(A), y::TrackedArray{V,D}) where {V,D} = record_plus(x, y, D)
end

@inline Base.:+(x::TrackedArray{V,D}, y::StaticArray) where {V,D} = record_plus(x, Array(y), D)
@inline Base.:+(x::StaticArray, y::TrackedArray{V,D}) where {V,D} = record_plus(Array(x), y, D)

function record_plus(x, y, ::Type{D}) where D
    tp = tape(x, y)
    out = track(value(x) + value(y), D, tp)
    record!(tp, SpecialInstruction, +, (x, y), out)
    return out
end

# reverse/forward passes

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(+)})
    a, b = instruction.input
    output = instruction.output
    output_deriv = deriv(output)
    istracked(a) && increment_deriv!(a, output_deriv)
    istracked(b) && increment_deriv!(b, output_deriv)
    unseed!(output)
    return nothing
end

@noinline function special_forward_exec!(instruction::SpecialInstruction{typeof(+)})
    a, b = instruction.input
    pull_value!(a)
    pull_value!(b)
    plus!(value(instruction.output), value(a), value(b))
    return nothing
end

###################
# subtraction (-) #
###################

# in-place version (necessary for nested differentiation to work)

function minus!(out, x)
    for i in eachindex(out)
        out[i] = -(x[i])
    end
end

function minus!(out, x, y)
    for i in eachindex(out)
        out[i] = x[i] - y[i]
    end
end

@inline minus!(out::TrackedArray, x::TrackedArray, y::TrackedArray) = record_minus!(out, x, y)
@inline minus!(out::TrackedArray, x::TrackedArray) = record_minus!(out, x, y)

for A in ARRAY_TYPES
    @eval @inline minus!(out::TrackedArray, x::TrackedArray, y::$(A)) = record_minus!(out, x, y)
    @eval @inline minus!(out::TrackedArray, x::$(A), y::TrackedArray) = record_minus!(out, x, y)
    @eval @inline minus!(out::TrackedArray, x::$(A)) = record_minus!(out, x)
end

function record_minus!(out::TrackedArray, x)
    copyto!(value(out), -(value(x)))
    record!(tape(x), SpecialInstruction, -, x, out)
    return out
end

function record_minus!(out::TrackedArray, x, y)
    copyto!(value(out), value(x) - value(y))
    record!(tape(x, y), SpecialInstruction, -, (x, y), out)
    return out
end

# Base allocating version

Base.:-(x::TrackedArray{X,D}, y::TrackedArray{Y,D}) where {X,Y,D} = record_minus(x, y, D)

for A in ARRAY_TYPES
    @eval Base.:-(x::TrackedArray{V,D}, y::$(A)) where {V,D} = record_minus(x, y, D)
    @eval Base.:-(x::$(A), y::TrackedArray{V,D}) where {V,D} = record_minus(x, y, D)
end

@inline Base.:-(x::TrackedArray{V,D}, y::StaticArray) where {V,D} = record_minus(x, Array(y), D)
@inline Base.:-(x::StaticArray, y::TrackedArray{V,D}) where {V,D} = record_minus(Array(x), y, D)

function Base.:-(x::TrackedArray{V,D}) where {V,D}
    tp = tape(x)
    out = track(-(value(x)), D, tp)
    record!(tp, SpecialInstruction, -, x, out)
    return out
end

function record_minus(x, y, ::Type{D}) where D
    tp = tape(x, y)
    out = track(value(x) - value(y), D, tp)
    record!(tp, SpecialInstruction, -, (x, y), out)
    return out
end

# reverse/forward passes

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(-)})
    input = instruction.input
    output = instruction.output
    output_deriv = deriv(output)
    if istracked(input)
        decrement_deriv!(input, output_deriv)
    else
        a, b = input
        istracked(a) && increment_deriv!(a, output_deriv)
        istracked(b) && decrement_deriv!(b, output_deriv)
    end
    unseed!(output)
    return nothing
end

@noinline function special_forward_exec!(instruction::SpecialInstruction{typeof(-)})
    input = instruction.input
    output = instruction.output
    if istracked(input)
        minus!(value(output), value(input))
    else
        a, b = input
        pull_value!(a)
        pull_value!(b)
        minus!(value(output), value(a), value(b))
    end
    return nothing
end

######################
# multiplication (*) #
######################

mulargvalue(x) = value(x)
mulargvalue(x::Adjoint) = adjoint(value(adjoint(x)))
mulargvalue(x::Transpose) = transpose(value(transpose(x)))

mulargpullvalue!(x) = pull_value!(x)
mulargpullvalue!(x::Adjoint) = pull_value!(adjoint(x))
mulargpullvalue!(x::Transpose) = pull_value!(transpose(x))

# recording pass #
#----------------#

@inline function record_mul(x, y, ::Type{D}) where D
    tp = tape(x, y)
    out = track(*(mulargvalue(x), mulargvalue(y)), D, tp)
    cache = (similar(x, D), similar(y, D))
    record!(tp, SpecialInstruction, *, (x, y), out, cache)
    return out
end

@inline function record_mul!(out::TrackedArray{V,D}, x, y) where {V,D}
    copyto!(mulargvalue(out), *(mulargvalue(x), mulargvalue(y)))
    cache = (similar(x, D), similar(y, D))
    record!(tape(x, y), SpecialInstruction, *, (x, y), out, cache)
    return out
end

for S1 in (:TrackedArray, :TrackedVector, :TrackedMatrix)
    for S2 in (:TrackedArray, :TrackedVector, :TrackedMatrix)
        @eval begin
            LinearAlgebra.:*(x::$(S1){X,D}, y::$(S2){Y,D}) where {X,Y,D} = record_mul(x, y, D)

            LinearAlgebra.:*(x::Transpose{<:TrackedReal,<:$(S1){X,D}}, y::Transpose{<:TrackedReal,<:$(S2){Y,D}}) where {X,Y,D} = record_mul(x, y, D)
            LinearAlgebra.:*(x::Adjoint{<:TrackedReal,<:$(S1){X,D}}, y::Adjoint{<:TrackedReal,<:$(S2){Y,D}}) where {X,Y,D} = record_mul(x, y, D)

            LinearAlgebra.:*(x::Transpose{<:TrackedReal,<:$(S1){X,D}}, y::$(S2){Y,D}) where {X,Y,D} = record_mul(x, y, D)
            LinearAlgebra.:*(x::$(S1){X,D}, y::Transpose{<:TrackedReal,<:$(S2){Y,D}}) where {X,Y,D} = record_mul(x, y, D)
            LinearAlgebra.:*(x::Adjoint{<:TrackedReal,<:$(S1){X,D}}, y::$(S2){Y,D}) where {X,Y,D} = record_mul(x, y, D)
            LinearAlgebra.:*(x::$(S1){X,D}, y::Adjoint{<:TrackedReal,<:$(S2){Y,D}}) where {X,Y,D} = record_mul(x, y, D)

            LinearAlgebra.mul!(out::TrackedArray{V,D}, x::$(S1){X,D}, y::$(S2){Y,D}) where {V,X,Y,D} = record_mul!(out, x, y)

            LinearAlgebra.mul!(out::TrackedArray{V,D}, x::Transpose{<:TrackedReal,<:$(S1){X,D}}, y::Transpose{<:TrackedReal,<:$(S2){Y,D}}) where {V,X,Y,D} = record_mul!(out, x, y)
            LinearAlgebra.mul!(out::TrackedArray{V,D}, x::Adjoint{<:TrackedReal,<:$(S1){X,D}}, y::Adjoint{<:Number,<:$(S2){Y,D}}) where {V,X,Y,D} = record_mul!(out, x, y)

            LinearAlgebra.mul!(out::TrackedArray{V,D}, x::Transpose{<:TrackedReal,<:$(S1){X,D}}, y::$(S2){Y,D}) where {V,X,Y,D} = record_mul!(out, x, y)
            LinearAlgebra.mul!(out::TrackedArray{V,D}, x::$(S1){X,D}, y::Transpose{<:TrackedReal,<:$(S2){Y,D}}) where {V,X,Y,D} = record_mul!(out, x, y)
            LinearAlgebra.mul!(out::TrackedArray{V,D}, x::Adjoint{<:TrackedReal,<:$(S1){X,D}}, y::$(S2){Y,D}) where {V,X,Y,D} = record_mul!(out, x, y)
            LinearAlgebra.mul!(out::TrackedArray{V,D}, x::$(S1){X,D}, y::Adjoint{<:TrackedReal,<:$(S2){Y,D}}) where {V,X,Y,D} = record_mul!(out, x, y)
        end
    end

    for T in ARRAY_TYPES
        @eval begin
            LinearAlgebra.:*(x::$(S1){V,D}, y::$(T)) where {V,D} = record_mul(x, y, D)
            LinearAlgebra.:*(x::$(T), y::$(S1){V,D}) where {V,D} = record_mul(x, y, D)

            LinearAlgebra.:*(x::Transpose{<:Number,<:$(T)}, y::$(S1){V,D}) where {V,D} = record_mul(x, y, D)
            LinearAlgebra.:*(x::$(S1){V,D}, y::Transpose{<:Number,<:$(T)}) where {V,D} = record_mul(x, y, D)
            LinearAlgebra.:*(x::Adjoint{<:Number,<:$(T)}, y::$(S1){V,D}) where {V,D} = record_mul(x, y, D)
            LinearAlgebra.:*(x::$(S1){V,D}, y::Adjoint{<:Number,<:$(T)}) where {V,D} = record_mul(x, y, D)

            LinearAlgebra.:*(x::Transpose{<:Number,<:$(S1){V,D}}, y::$(T)) where {V,D} = record_mul(x, y, D)
            LinearAlgebra.:*(x::$(T), y::Transpose{<:Number,<:$(S1){V,D}}) where {V,D} = record_mul(x, y, D)
            LinearAlgebra.:*(x::Adjoint{<:Number,<:$(S1){V,D}}, y::$(T)) where {V,D} = record_mul(x, y, D)
            LinearAlgebra.:*(x::$(T), y::Adjoint{<:Number,<:$(S1){V,D}}) where {V,D} = record_mul(x, y, D)

            LinearAlgebra.mul!(out::TrackedArray, x::$(S1), y::$(T)) = record_mul!(out, x, y)
            LinearAlgebra.mul!(out::TrackedArray, x::$(T), y::$(S1)) = record_mul!(out, x, y)

            LinearAlgebra.mul!(out::TrackedArray, x::$(S1), y::Transpose{<:Number,<:$(T)}) = record_mul!(out, x, y)
            LinearAlgebra.mul!(out::TrackedArray, x::Transpose{<:Number,<:$(T)}, y::$(S1)) = record_mul!(out, x, y)
            LinearAlgebra.mul!(out::TrackedArray, x::$(S1), y::Adjoint{<:Number,<:$(T)}) = record_mul!(out, x, y)
            LinearAlgebra.mul!(out::TrackedArray, x::Adjoint{<:Number,<:$(T)}, y::$(S1)) = record_mul!(out, x, y)

            LinearAlgebra.mul!(out::TrackedArray, x::Transpose{<:Number,<:$(S1)}, y::$(T)) = record_mul!(out, x, y)
            LinearAlgebra.mul!(out::TrackedArray, x::$(T), y::Transpose{<:Number,<:$(S1)}) = record_mul!(out, x, y)
            LinearAlgebra.mul!(out::TrackedArray, x::Adjoint{<:Number,<:$(S1)}, y::$(T)) = record_mul!(out, x, y)
            LinearAlgebra.mul!(out::TrackedArray, x::$(T), y::Adjoint{<:Number,<:$(S1)}) = record_mul!(out, x, y)
        end
    end
end

# forward pass #
#--------------#

@noinline function special_forward_exec!(instruction::SpecialInstruction{typeof(*)})
    a, b = instruction.input
    mulargpullvalue!(a)
    mulargpullvalue!(b)
    if instruction.output isa Number
        value!(instruction.output, mulargvalue(a) * mulargvalue(b))
    else
        mul!(mulargvalue(instruction.output), mulargvalue(a), mulargvalue(b))
    end
    return nothing
end

# reverse pass #
#--------------#

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(*)})
    a, b = instruction.input
    a_tmp, b_tmp = instruction.cache
    output = instruction.output
    output_deriv = deriv(output)
    reverse_mul!(output, output_deriv, a, b, a_tmp, b_tmp)
    unseed!(output)
    return nothing
end

# a * b

function reverse_mul!(output, output_deriv, a, b, a_tmp, b_tmp)
    istracked(a) && increment_deriv!(a, mul!(a_tmp, output_deriv, transpose(value(b))))
    istracked(b) && increment_deriv!(b, mul!(b_tmp, transpose(value(a)), output_deriv))
end

for (f, F) in ((:transpose, :Transpose), (:adjoint, :Adjoint))
    @eval begin
        # a * f(b)
        function reverse_mul!(output, output_deriv, a, b::$F, a_tmp, b_tmp)
            _b = ($f)(b)
            istracked(a) && increment_deriv!(a, mul!(a_tmp, output_deriv, mulargvalue(b)))
            istracked(_b) && increment_deriv!(_b, ($f)(mul!(b_tmp, ($f)(output_deriv), value(a))))
        end
           # f(a) * b
        function reverse_mul!(output, output_deriv, a::$F, b, a_tmp, b_tmp)
            _a = ($f)(a)
            istracked(_a) && increment_deriv!(_a, ($f)(mul!(a_tmp, output_deriv, value(b))))
            istracked(b) && increment_deriv!(b, mul!(b_tmp, ($f)(mulargvalue(a)), ($f)(output_deriv)))
        end
        # f(a) * f(b)
        function reverse_mul!(output, output_deriv, a::$F, b::$F, a_tmp, b_tmp)
            _a = ($f)(a)
            _b = ($f)(b)
            istracked(_a) && increment_deriv!(_a, ($f)(mul!(a_tmp, ($f)(mulargvalue(b)), ($f)(output_deriv))))
            istracked(_b) && increment_deriv!(_b, ($f)(mul!(b_tmp, ($f)(output_deriv), ($f)(mulargvalue(a)))))
        end
    end
end

# adjoint(a) * transpose(b)

function reverse_mul!(output, output_deriv, a::Adjoint, b::Transpose, a_tmp, b_tmp)
    _a = adjoint(a)
    _b = transpose(b)
    if istracked(_a)
        reverse_mul!(output, output_deriv, transpose(_a), b, a_tmp, b_tmp)
    elseif istracked(_b)
        increment_deriv!(_b, transpose(mul!(b_tmp, adjoint(output_deriv), adjoint(mulargvalue(a)))))
    end
end

# transpose(a) * adjoint(b)

function reverse_mul!(output, output_deriv, a::Transpose, b::Adjoint, a_tmp, b_tmp)
    _a = transpose(a)
    _b = adjoint(b)
    if istracked(_b)
        reverse_mul!(output, output_deriv, a, transpose(_b), a_tmp, b_tmp)
    elseif istracked(_a)
        increment_deriv!(_a, transpose(mul!(a_tmp, adjoint(mulargvalue(b)), adjoint(output_deriv))))
    end
end

## zero

Base.zero(x::ReverseDiff.TrackedArray) = track(zero(x.value))
