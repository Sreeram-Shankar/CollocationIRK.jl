const _TABLEAU_CACHE = Dict{Tuple{Symbol,Int}, NamedTuple}()
empty!(_TABLEAU_CACHE)

#gets the tableau from the cache or builds it if it is not in the cache
function get_tableau(family::Symbol, s::Int, verbose::Bool=false)
    key = (family, s)
    haskey(_TABLEAU_CACHE, key) && return _TABLEAU_CACHE[key]

    if family == :gl
        A, b, c = build_gauss_legendre_irk(s, verbose)
        entry = (A=A, b=b, c=c)
    elseif family == :radau
        A, b, c = build_radau_irk(s, verbose)

        #perform a Schur decomposition to make the tableau diagonally dominant
        if verbose
            println("Performing Schur decomposition")
        end
        A_f64 = Matrix{Float64}(A)
        F = schur(A_f64)
        if verbose
            println("Schur decomposition complete")
        end
        entry = (A=A, b=b, c=c, Z=F.Z, T=F.T)
    else
        error("Unknown family $family, must be :gl or :radau")
    end

    _TABLEAU_CACHE[key] = entry
    return entry
end