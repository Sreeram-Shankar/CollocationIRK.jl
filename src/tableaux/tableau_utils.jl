#formats the number string still in bigfloat form
function nstr_fixed(x::BigFloat, digits::Int=80)
    s = string(BigFloat(round(x, digits=digits)))
    if occursin(".", s)
        s = rstrip(s, '0')
        s = rstrip(s, '.')
    end
    return s
end


#builds the Lagrange basis polynomial when building the butcher tableau
function lagrange_basis(c::Vector{BigFloat}, j::Int)
    xj = c[j]
    others = [c[k] for k in eachindex(c) if k != j]
    
    denom = one(BigFloat)
    for xk in others
        denom *= (xj - xk)
    end
    
    function Lj(x::BigFloat)
        num = one(BigFloat)
        for xk in others
            num *= (x - xk)
        end
        return num / denom
    end
    
    return Lj
end


#writes the butcher tableau to a file formatted as triplets (i, j, A[i, j])
function write_A_b_c_triplets(A::Matrix{BigFloat}, b::Vector{BigFloat}, c::Vector{BigFloat}, basename::String, digits::Int=80)
    s = length(b)
    open("$(basename)_triplets.txt", "w") do ft
        for i in 1:s
            for j in 1:s
                println(ft, "$i $j $(nstr_fixed(A[i, j], digits))")
            end
        end
        for i in 1:s
            println(ft, "$i 0 $(nstr_fixed(c[i], digits))")
        end
        for j in 1:s
            println(ft, "0 $j $(nstr_fixed(b[j], digits))")
        end
    end
end


#implements the tableau correctness checks, specifically row-sum and moment checks
function check_tableau(A::Matrix{BigFloat}, b::Vector{BigFloat}, c::Vector{BigFloat}, verbose::Bool)
    if verbose
        println("\nRow-sum checks (ΣA[i,:] ≈ c[i]):")
    end
    for i in eachindex(c)
        ssum = sum(A[i, :])
        diff = ssum - c[i]
        if verbose
            println("  i=$i: ΣA=$ssum  c=$(c[i])  diff=$diff")
        end
    end
    if verbose
        println("\nMoment checks (b·c^k ≈ 1/(k+1)):")
    end
    for k in 0:min(9, 2*length(c)-1)
        lhs = sum(b[j] * c[j]^k for j in eachindex(c))
        rhs = BigFloat(1) / (k + 1)
        err = lhs - rhs
        if verbose
            println("  k=$k: $lhs  target=$rhs  err=$err")
        end
    end
end