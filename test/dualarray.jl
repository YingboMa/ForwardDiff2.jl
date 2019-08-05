using ForwardDiff2: DualArray, tagname

@testset "DualArray" begin
    a = rand(2, 2)
    d1 = DualArray(a)
    tagname(d1) === Nothing
    d2 = DualArray{2}(a)
    tagname(d2) === 2
end
