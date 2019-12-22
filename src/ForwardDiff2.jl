module ForwardDiff2

export DualArray, jacobian

include("custom_dispatch.jl")
include("tag.jl")
include("dualarray.jl")
include("dual_context.jl")
include("jacobian.jl")

# Experimental
#include("aosoa.jl")

end # module
