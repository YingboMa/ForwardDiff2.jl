using DifferentialEquations, DiffEqSensitivity, ForwardDiff2
using ForwardDiff2: D, Dual, DualArray

function test()
    function f(u,p,t)
        dx = p[1]*u[1] - p[2]*u[1]*u[2]
        dy = -p[3]*u[2] + u[1]*u[2]
        [dx, dy]
    end

    p = [1.5,1.0,3.0,1.0]
    u0 = [1.0;1.0]
    prob = ODEProblem(f,u0,(0.0,10.0),p)
    function test_f(p)
        t = convert.(eltype(p), u0)
        _prob = remake(prob;u0=t,p=p)
        solve(_prob,Vern9(),save_everystep=false)[end]
    end
end

"""
If a is a DualArray,
Copy b to be a DualArray similar to a
"""
function similarcopy(a::DualArray, b)
    pa = ForwardDiff2.allpartials(a)
    DualArray(copy(b),
              zeros(eltype(b),
                    (size(b)..., size(pa,ndims(a)+1))))
end

function similarcopy(a,b)
    return b
end
