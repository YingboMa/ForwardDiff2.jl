module ForwardDiff2

export DualArray, jacobian

include("dualarray.jl")
include("jacobian.jl")
# TODO: differentiation API (jacobian...)

end # module
