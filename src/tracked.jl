#########
# Types #
#########

const NULL_INDEX = typemin(Int)
const NULL_TAPE = InstructionTape()

# TrackedReal #
#-------------#

#=
A `TrackedReal` stores a value and a reference back to the original `TrackedArray` which
provided the value.

When performing a forward pass through a previously-recorded `InstructionTape`, any encountered
`TrackedReal` instances which are direct descendents of their origin array must re-validate
themselves by re-retrieving their value from their origin via the given index. A similar
strategy is taken during the reverse pass, when derivs are updated; they are always
re-validated with the origin before and after "local" updates to the TrackedReal. The big
benefit of this strategy is that scalar `getindex` operations don't need to be explicitly
recorded to the tape.

Note that we don't have to worry about the origin's values being invalidated during `InstructionTape`
execution, since the `TrackedArray` type is immutable.

Also note that it's possible to instantiate an origin-less `TrackedReal`. This will most
often happen as a result of conversion, or via a numerical constructor (e.g. `zero`).
In that case, we leave the `origin` field uninitialized, and set the `index` field to
`NULL_INDEX`.

There are two alternative approaches we could've taken for dealing with the potential
missing-ness of the origin, and it's worth explaining why we don't take these approaches.

The first alternative is to encode the lack of an origin as a parameter in the
`TrackedReal` type. Indeed, this is exactly what we do if the origin of a `TrackedReal` is a
constructor like `zero`, setting `O` to `Nothing`. However, this strategy is not a viable
solution for origin-less `TrackedReal`s that result from conversions. For example, take the
expression `y = convert(::Type{T<:TrackedReal}, x::Real)`. `y` will not have an origin, but
`isa(y, T)` MUST hold true regardless. This could break in a world where `T` forcibly
specifies an origin of type `O != Nothing`.

The second alternative is to use `Nullable` + `isnull`. This solution would work, but it
overcomplicates the API and would incur unneccesary pointer loads during re-validation,
making the below implementation preferable.
=#

mutable struct TrackedReal{V<:Real,D<:Real,O} <: Real
    value::V
    deriv::D
    tape::InstructionTape
    index::Int
    origin::O
    TrackedReal{V,D,O}(value, deriv, tape, index, origin) where {V,D,O} = new{V,D,O}(value, deriv, tape, index, origin)
    TrackedReal{V,D,O}(value, deriv, tape) where {V,D,O} = new{V,D,O}(value, deriv, tape, NULL_INDEX)
    TrackedReal{V,D,O}(value, deriv) where {V,D,O} = new{V,D,O}(value, deriv, NULL_TAPE, NULL_INDEX)
    TrackedReal{V,D,O}(value) where {V,D,O} = new{V,D,O}(value, zero(D), NULL_TAPE, NULL_INDEX)
end

TrackedReal(v::V, a::D, tp::InstructionTape, i::Int, o::O) where {V,D,O} = TrackedReal{V,D,O}(v, a, tp, i, o)

TrackedReal(v::V, a::D, tp::InstructionTape = NULL_TAPE) where {V,D} = TrackedReal{V,D,Nothing}(v, a, tp)

# we define these special cases so that the "constructor <--> convert" pun holds for `TrackedReal`
# this is Jarett's favorite piece of code. A true work of art.
@inline TrackedReal{V,D,O}(x::TrackedReal) where {V,D,O} = convert(TrackedReal{V,D,O}, x)

# TrackedArray #
#--------------#

struct TrackedArray{V,D,N,VA,DA} <: AbstractArray{TrackedReal{V,D,TrackedArray{V,D,N,VA,DA}},N}
    value::VA
    deriv::DA
    tape::InstructionTape
    function TrackedArray{V,D,N,VA,DA}(value::AbstractArray{V,N},
                                       deriv::AbstractArray{D,N},
                                       tape::InstructionTape) where {V,D,N,VA,DA}
        @assert IndexStyle(value) === IndexLinear()
        @assert size(value) === size(deriv)
        return new{V,D,N,VA,DA}(value, deriv, tape)
    end
end

function TrackedArray(value::AbstractArray{V,N},
                      deriv::AbstractArray{D,N},
                      tape::InstructionTape) where {V,D,N}
    return TrackedArray{V,D,N,typeof(value),typeof(deriv)}(value, deriv, tape)
end

const TrackedVector{V,D} = TrackedArray{V,D,1}
const TrackedMatrix{V,D} = TrackedArray{V,D,2}
const TrackedVecOrMat{V,D} = Union{TrackedVector{V,D}, TrackedMatrix{V,D}}

###########
# getters #
###########

istracked(x) = false
istracked(::TrackedReal) = true
istracked(::TrackedArray) = true
istracked(::AbstractArray{T}) where {T} = T <: TrackedReal || !(isconcretetype(T))

@inline value(x) = x
@inline value(x::AbstractArray) = istracked(x) ? map(value, x) : x
@inline value(t::TrackedReal) = t.value
@inline value(t::TrackedArray) = t.value

@inline deriv(t::TrackedArray) = t.deriv
@inline deriv(t::TrackedReal) =  t.deriv

@inline valtype(::TrackedReal{V}) where {V} = V
@inline valtype(::Type{TrackedReal{V,D,O}}) where {V,D,O} = V
@inline valtype(::TrackedArray{V}) where {V} = V
@inline valtype(::Type{TrackedArray{V,D,N,VA,DA}}) where {V,D,VA,DA,N} = V

@inline derivtype(::TrackedReal{V,D}) where {V,D} = D
@inline derivtype(::Type{TrackedReal{V,D,O}}) where {V,D,O} = D
@inline derivtype(t::TrackedArray{V,D}) where {V,D} = D
@inline derivtype(::Type{TrackedArray{V,D,N,VA,DA}}) where {V,D,VA,DA,N} = D

@inline origintype(::TrackedReal{V,D,O}) where {V,D,O} = O
@inline origintype(::Type{TrackedReal{V,D,O}}) where {V,D,O} = O

@inline hasorigin(x::Real) = false
@inline hasorigin(t::TrackedReal) = t.index !== NULL_INDEX

@inline hastape(x) = false
@inline hastape(t::TrackedArray) = tape(t) !== NULL_TAPE
@inline hastape(t::TrackedReal) = tape(t) !== NULL_TAPE
@inline hastape(x::AbstractArray) = tape(x) !== NULL_TAPE

@inline tape(x) = NULL_TAPE
@inline tape(t::TrackedArray) = t.tape
@inline tape(t::TrackedReal) = t.tape

function tape(x::AbstractArray)
    if istracked(x)
        for i in x
            hastape(i) && return tape(i)
        end
    end
    return NULL_TAPE
end

function tape(ts...)
    for t in ts
        hastape(t) && return tape(t)
    end
    return NULL_TAPE
end

###########
# setters #
###########

@inline value!(t::TrackedReal, v::Real) = (t.value = v; nothing)
@inline value!(t::TrackedArray, v::AbstractArray) = (copyto!(value(t), v); nothing)

function value!(t::NTuple{N,Any}, v::NTuple{N,Any}) where N
    for i in eachindex(t)
        value!(t[i], v[i])
    end
    return nothing
end

@inline deriv!(t::TrackedReal, v::Real) = (t.deriv = v; nothing)
@inline deriv!(t::TrackedArray, v::AbstractArray) = (copyto!(deriv(t), v); nothing)

function deriv!(t::NTuple{N,Any}, v::NTuple{N,Any}) where N
    for i in eachindex(t)
        deriv!(t[i], v[i])
    end
    return nothing
end

deriv!(t::StaticArray, v::AbstractArray) = deriv!(Tuple(t), Tuple(v))

# pulling values from origin #
#----------------------------#

pull_value!(x) = nothing
pull_value!(t::TrackedArray) = nothing
pull_value!(t::TrackedReal) = (hasorigin(t) && value!(t, value(t.origin)[t.index]); nothing)
pull_value!(x::AbstractArray) = (istracked(x) && foreach(pull_value!, x); nothing)

# pulling derivs from origin #
#----------------------------#

pull_deriv!(x) = nothing
pull_deriv!(t::TrackedArray) = nothing
pull_deriv!(t::TrackedReal) = (hasorigin(t) && deriv!(t, deriv(t.origin)[t.index]); nothing)
pull_deriv!(x::AbstractArray) = (istracked(x) && foreach(pull_deriv!, x); nothing)

# pushing derivs back to origin #
#-------------------------------#

push_deriv!(x) = nothing
push_deriv!(t::TrackedArray) = nothing
push_deriv!(t::TrackedReal) = (hasorigin(t) && (t.origin.deriv[t.index] = deriv(t)); nothing)
push_deriv!(x::AbstractArray) = (istracked(x) && foreach(push_deriv!, x); nothing)

# seed/unseed #
#-------------#

seed!(x) = nothing
seed!(t::TrackedReal) = (t.deriv = one(derivtype(t)); push_deriv!(t); nothing)
seed!(t::TrackedArray, i) = (t.deriv[i] = one(derivtype(t)); nothing)
seed!(x::AbstractArray, i) = seed!(x[i])

unseed!(x) = nothing
unseed!(t::TrackedReal) = (t.deriv = zero(derivtype(t)); push_deriv!(t); nothing)
unseed!(t::TrackedArray) = (fill!(deriv(t), zero(derivtype(t))); nothing)
unseed!(x::AbstractArray) = (istracked(x) && foreach(unseed!, x); nothing)
unseed!(t::Tuple) = foreach(unseed!, t)
unseed!(t::TrackedArray, i) = (t.deriv[i] = zero(derivtype(t)); nothing)
unseed!(x::AbstractArray, i) = unseed!(x[i])

#########################
# capture (see tape.jl) #
#########################

# This is type unstable, but that shouldn't be too much of a problem as it's only used below
# the function barrier of `record!`, which just pushes `t` to the tape and always returns
# `nothing`. It enables us to not waste time updating constants during
# `forward_pass!`/`reverse_pass!`.
capture(t::TrackedReal) = ifelse(hastape(t), t, value(t))
capture(t::TrackedArray) = t
capture(t::AbstractArray) = istracked(t) ?  map!(capture, similar(t), t) : copy(t)
# `StaticArray`s don't support mutation unless the eltype is a bits type (`isbitstype`).
capture(t::SA) where SA <: StaticArray = istracked(t) ? SA(map(capture, t)) : copy(t)

########################
# Conversion/Promotion #
########################

_convert(::Type{R}, t::TrackedReal) where {R <: Real} = R(value(t))
_convert(::Type{R}, t::Real) where {R <: TrackedReal} = R(t)
_convert(::Type{R}, t::Real) where {R <: Real} = R(t)
_convert(::Type{R}, t::TrackedReal) where {R <: TrackedReal} = R(t)

# recording a instruction for this preserves the line of references back to the origin's deriv
function Base.convert(::Type{T1}, t::T2) where {T1<:TrackedReal,T2<:TrackedReal}
    V1, D1, O1 = valtype(T1), derivtype(T1), origintype(T1)
    tp = tape(t)
    out = TrackedReal{V1,D1,O1}(_convert(V1, value(t)), _convert(D1, deriv(t)), tp)
    record!(tp, SpecialInstruction, convert, t, out)
    return out
end

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(convert)})
    output = instruction.output
    increment_deriv!(instruction.input, deriv(output))
    unseed!(output)
    return nothing
end

@noinline function special_forward_exec!(instruction::SpecialInstruction{typeof(convert)})
    input = instruction.input
    pull_value!(input)
    value!(instruction.output, value(input))
    return nothing
end

Base.convert(::Type{Real}, t::T) where {T<:TrackedReal} = t
function Base.convert(::Type{R}, t::T) where {R<:Real,T<:TrackedReal}
    throw(ArgumentError("Converting an instance of $T to $R is not defined. Please use `ReverseDiff.value` instead."))
end
function Base.convert(::Type{T}, x::R) where {T<:TrackedReal, R<:Real}
    return TrackedReal{valtype(T), derivtype(T), origintype(T)}(
        convert(valtype(T), value(x)),
    )
end

Base.convert(::Type{T}, t::T) where {T<:TrackedReal} = t
Base.convert(::Type{T}, t::T) where {T<:TrackedArray} = t

for R in REAL_TYPES
    @eval Base.promote_rule(::Type{$R}, ::Type{TrackedReal{V,D,O}}) where {V,D,O} = TrackedReal{promote_type($R,V),D,O}
end

Base.promote_rule(::Type{R}, ::Type{TrackedReal{V,D,O}}) where {R<:Real,V,D,O} = TrackedReal{promote_type(R,V),D,O}
Base.promote_rule(::Type{TrackedReal{V1,D1,O1}}, ::Type{TrackedReal{V2,D2,O2}}) where {V1,V2,D1,D2,O1,O2} = TrackedReal{promote_type(V1,V2),promote_type(D1,D2),Nothing}

###########################
# AbstractArray Interface #
###########################

Base.getindex(t::TrackedArray, i::Int) = TrackedReal(value(t)[i], deriv(t)[i], tape(t), i, t)

colon2range(s, i) = i
colon2range(s, ::Colon) = s

function index_iterable(shape::NTuple{N,Any}, i::NTuple{M,Any}) where {N,M}
    if N < M
        return index_iterable(shape, ntuple(n -> i[n], Val(N)))
    elseif M < N && isa(last(i), Colon)
        return index_iterable(shape, ntuple(n -> (n > M ? Colon() : i[n]), Val(N)))
    else
        return Base.Iterators.product(map(colon2range, shape[1:M], i)...)
    end
end

for T in (:AbstractRange, :Colon, :(Union{Colon,AbstractRange}))
    @eval function Base.getindex(t::TrackedArray, i::$(T)...)
        tp = tape(t)
        out = TrackedArray(value(t)[i...], deriv(t)[i...], tp)
        idx = index_iterable(axes(t), i)
        record!(tp, SpecialInstruction, getindex, (t, idx), out)
        return out
    end
end

@noinline function special_reverse_exec!(instruction::SpecialInstruction{typeof(getindex)})
    input, indices = instruction.input
    output = instruction.output
    input_deriv = deriv(input)
    output_deriv = deriv(output)
    i = 0
    for idx in indices
        input_deriv[CartesianIndex(idx)] += output_deriv[i += 1]
    end
    unseed!(output)
    return nothing
end

@noinline function special_forward_exec!(instruction::SpecialInstruction{typeof(getindex)})
    input, indices = instruction.input
    input_value = value(input)
    output_value = value(instruction.output)
    i = 0
    for idx in indices
        output_value[i += 1] = input_value[CartesianIndex(idx)]
    end
    return nothing
end

function Base.getindex(t::TrackedArray, inds::AbstractArray{<:CartesianIndex})
    tp = tape(t)
    out = TrackedArray(value(t)[inds], deriv(t)[inds], tp)
    record!(tp, SpecialInstruction, getindex, (t, inds), out)
    return out
end
function Base.getindex(t::TrackedArray, i::Int...)
    ind = LinearIndices(t)[i...]
    return TrackedReal(value(t)[i...], deriv(t)[i...], tape(t), ind, t)
end
function Base.getindex(t::TrackedArray, _inds::Union{Integer, Colon, AbstractArray{<:Integer}}...)
    inds = ntuple(Val(length(_inds))) do i
        _inds[i] isa Colon && return firstindex(t,i):lastindex(t,i)
        return _inds[i]
    end
    tp = tape(t)
    out = TrackedArray(value(t)[inds...], deriv(t)[inds...], tp)
    record!(tp, SpecialInstruction, (getindex, Val(:generic)), (t, inds), out)
    return out
end
@noinline function special_reverse_exec!(instruction::SpecialInstruction{<:Tuple{typeof(getindex), Val{:generic}}})
    input, inds = instruction.input
    output = instruction.output
    cinds = CartesianIndices(map(i -> 1:length(i), inds))
    input_deriv = deriv(input)
    output_deriv = deriv(output)
    i = 0
    for _idx in cinds
        idx = CartesianIndex(map(getindex, inds, Tuple(_idx)))
        input_deriv[idx] += output_deriv[i += 1]
    end
    unseed!(output)
    return nothing
end
@noinline function special_forward_exec!(instruction::SpecialInstruction{<:Tuple{typeof(getindex), Val{:generic}}})
    input, inds = instruction.input
    input_value = value(input)
    output_value = value(instruction.output)
    cinds = CartesianIndices(map(i -> 1:length(i), inds))
    i = 0
    for cind in cinds
        idx = CartesianIndex(map(getindex, inds, Tuple(cind)))
        output_value[i += 1] = input_value[idx]
    end
    return nothing
end

Base.setindex!(t::TrackedArray, args...) = error("TrackedArrays do not support setindex!")

Base.IndexStyle(::TrackedArray) = IndexLinear()

Base.size(t::TrackedArray) = size(value(t))

Base.copy(t::TrackedArray) = t

Base.similar(t::TrackedArray, args::Union{Integer, AbstractUnitRange}...) = similar(value(t), eltype(t), args...)

Base.similar(t::TrackedArray, T::Type, args::Union{Integer, AbstractUnitRange}...) = similar(value(t), T, args...)

reshape_body = :(TrackedArray(reshape(value(t), dims), reshape(deriv(t), dims), tape(t)))
@eval Base.reshape(t::TrackedArray, dims::Val{N}) where {N} = $reshape_body
@eval Base.reshape(t::TrackedArray, dims::Tuple{Vararg{Int,N}}) where {N} = $reshape_body
@eval Base.reshape(t::TrackedArray, dims::Int64...) = $reshape_body
@eval Base.reshape(t::TrackedArray, dims::AbstractUnitRange...) = $reshape_body
@eval Base.reshape(t::TrackedArray, dims::Colon...) = $reshape_body
@eval Base.reshape(t::TrackedArray, dims::Union{AbstractUnitRange,Int64,Colon}...) = $reshape_body

####################
# `Real` Interface #
####################

Base.hash(t::TrackedReal) = hash(value(t))
Base.hash(t::TrackedReal, hsh::UInt64) = hash(value(t), hsh)

Base.deepcopy(t::T) where {T<:TrackedReal} = t
Base.copy(t::T) where {T<:TrackedReal} = t

function Base.float(t::TrackedReal{V,D,O}) where {V,D,O}
    v = float(value(t))
    return TrackedReal{typeof(v),D,O}(v)
end

Base.float(t::TrackedReal{V}) where {V<:AbstractFloat} = t

Base.one(::Type{TrackedReal{V,D,O}}) where {V,D,O} = TrackedReal{V,D,O}(one(V))
Base.zero(::Type{TrackedReal{V,D,O}}) where {V,D,O} = TrackedReal{V,D,O}(zero(V))

Base.rand(::Type{TrackedReal{V,D,O}}) where {V,D,O} = TrackedReal{V,D,O}(rand(V))
Base.rand(rng::Random.AbstractRNG, ::Type{TrackedReal{V,D,O}}) where {V,D,O} = TrackedReal{V,D,O}(rand(rng, V))

Base.eps(t::TrackedReal) = eps(value(t))
Base.eps(::Type{T}) where {T<:TrackedReal} = eps(valtype(T))

Base.floor(t::TrackedReal) = floor(value(t))
Base.floor(::Type{R}, t::TrackedReal) where {R<:Real} = floor(R, value(t))

Base.ceil(t::TrackedReal) = ceil(value(t))
Base.ceil(::Type{R}, t::TrackedReal) where {R<:Real} = ceil(R, value(t))

Base.trunc(t::TrackedReal) = trunc(value(t))
Base.trunc(::Type{R}, t::TrackedReal) where {R<:Real} = trunc(R, value(t))

Base.round(t::TrackedReal) = round(value(t))
Base.round(::Type{R}, t::TrackedReal) where {R<:Real} = round(R, value(t))

Base.oneunit(t::TrackedReal) = one(t)
Base.oneunit(::Type{T}) where {T<:TrackedReal} = one(T)

Base.rtoldefault(::Type{T}) where {T<:TrackedReal} = sqrt(eps(T))

################
# track/track! #
################

track(x::Real, tp::InstructionTape = InstructionTape()) = track(x, typeof(x), tp)

track(x::AbstractArray, tp::InstructionTape = InstructionTape()) = track(x, eltype(x), tp)

track(x::Real, ::Type{D}, tp::InstructionTape = InstructionTape()) where {D} = TrackedReal(x, zero(D), tp)

track(x::AbstractArray, ::Type{D}, tp::InstructionTape = InstructionTape()) where {D} = TrackedArray(x, fill!(similar(x, D), zero(D)), tp)

track!(t::TrackedArray, x::AbstractArray) = (value!(t, x); unseed!(t); t)

track!(t::TrackedReal, x::Real) = (value!(t, x); unseed!(t); t)

function track!(t::AbstractArray{TrackedReal{D,D,Nothing}}, x::AbstractArray, tp::InstructionTape) where D
    for i in eachindex(t)
        t[i] = track(x[i], D, tp)
    end
    return t
end

###################
# Pretty Printing #
###################

idstr(x) = string(objectid(x), base=62)[1:3]

function Base.show(io::IO, t::TrackedReal)
    tape_id = hastape(t) ? idstr(t.tape) : "---"
    origin_id = hasorigin(t) ? "$(t.index), $(idstr(t.origin))" : "---"
    id = idstr(t)
    print(io, "TrackedReal<$(id)>($(value(t)), $(deriv(t)), $(tape_id), $(origin_id))")
end
