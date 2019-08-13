module ForwardDiff2

export DualArray, jacobian

include("dualarray.jl")
include("dual_context.jl")
include("jacobian.jl")

end # module
