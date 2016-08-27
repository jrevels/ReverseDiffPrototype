function dualwrap{S,N,T}(::Type{S}, duals::AbstractArray{Dual{N,T}}, tr::Nullable{Trace})
    ts = similar(duals, TraceReal{S,T})
    ps = similar(duals, Partials{N,T})
    for i in eachindex(duals)
        dual = duals[i]
        ts[i] = TraceReal{S}(value(dual), tr)
        ps[i] = partials(dual)
    end
    return ts, ps
end

for A in ARRAY_TYPES
    # map/broadcast on arrays #
    #-------------------------#
    for g in (:map, :broadcast)
        @eval begin
            function Base.$(g){F,S,T}(fopt::ForwardOptimize{F}, x::$(A){TraceReal{S,T}})
                fdual = t -> fopt.f(Dual(value(t), one(T)))
                duals = $(g)(fdual, x)
                tr = trace(x)
                out, partials = dualwrap(S, duals, tr)
                record!(tr, nothing, x, out, partials)
                return out
            end

            function Base.$(g){F,S,T1,T2}(fopt::ForwardOptimize{F},
                                          x1::$(A){TraceReal{S,T1}},
                                          x2::$(A){TraceReal{S,T2}})
                fdual = (t1, t2) -> fopt.f(Dual(value(t1), one(T1), zero(T1)),
                                           Dual(value(t2), zero(T2), one(T2)))
                duals = $(g)(fdual, x1, x2)
                tr = trace(x1, x2)
                out, partials = dualwrap(S, duals, tr)
                record!(tr, nothing, (x1, x2), out, partials)
                return out
            end
        end
    end

    # broadcast scalars vs. arrays #
    #------------------------------#
    for R in REAL_TYPES
        @eval begin
            @inline function Base.broadcast{F,S,T}(fopt::ForwardOptimize{F}, n::$R, x::$(A){TraceReal{S,T}})
                newf = ForwardOptimize(t -> fopt.f(n, t))
                return broadcast(newf, x)
            end

            @inline function Base.broadcast{F,S,T}(fopt::ForwardOptimize{F}, x::$(A){TraceReal{S,T}}, n::$R)
                newf = ForwardOptimize(t -> fopt.f(t, n))
                return broadcast(newf, x)
            end
        end
    end

    @eval begin
        function Base.broadcast{F,S,T,X}(fopt::ForwardOptimize{F}, n::TraceReal{S,T}, x::$(A){TraceReal{S,X}})
            ndual = Dual(value(n), one(T), zero(T))
            fdual = t -> fopt.f(ndual, Dual(value(t), zero(X), one(X)))
            duals = broadcast(fdual, x)
            tr = trace(n, x)
            out, partials = dualwrap(S, duals, tr)
            record!(tr, nothing, (n, x), out, partials)
            return out
        end

        function Base.broadcast{F,S,T,X}(fopt::ForwardOptimize{F}, x::$(A){TraceReal{S,X}}, n::TraceReal{S,T})
            ndual = Dual(value(n), zero(T), one(T))
            fdual = t -> fopt.f(Dual(value(t), one(X), zero(X)), ndual)
            duals = broadcast(fdual, x)
            tr = trace(n, x)
            out, partials = dualwrap(S, duals, tr)
            record!(tr, nothing, (x, n), out, partials)
            return out
        end
    end

    # standard elementwise operations (.+, .-, .*, etc.) #
    #----------------------------------------------------#
    for f in (:.+, :.-, :.*, :./, :.\, :.^)
        @eval begin
            @inline function Base.$(f){S,X,Y}(x::$(A){TraceReal{S,X}}, y::$(A){TraceReal{S,Y}})
                return broadcast(ForwardOptimize($(f)), x, y)
            end

            @inline function Base.$(f){S,T}(n::TraceReal, x::$(A){TraceReal{S,T}})
                return broadcast(ForwardOptimize($(f)), n, x)
            end

            @inline function Base.$(f){S,T}(x::$(A){TraceReal{S,T}}, n::TraceReal)
                return broadcast(ForwardOptimize($(f)), x, n)
            end
        end
        for R in REAL_TYPES
            @eval begin
                @inline function Base.$(f){S,T}(n::$R, x::$(A){TraceReal{S,T}})
                    return broadcast(ForwardOptimize($(f)), n, x)
                end

                @inline function Base.$(f){S,T}(x::$(A){TraceReal{S,T}}, n::$R)
                    return broadcast(ForwardOptimize($(f)), x, n)
                end
            end
        end
    end
end
