using F1Tenth
using Test

@testset "LatestChannel: put_latest! never blocks and overwrites a pending value" begin
    lc = F1Tenth.LatestChannel{Int}()
    F1Tenth.put_latest!(lc, 1)
    F1Tenth.put_latest!(lc, 2)
    @test F1Tenth.take_latest!(lc) === 2
end

@testset "LatestChannel: take_latest! returns missing on empty and drains exactly once" begin
    lc = F1Tenth.LatestChannel{Int}()
    @test F1Tenth.take_latest!(lc) === missing
    F1Tenth.put_latest!(lc, 5)
    @test F1Tenth.take_latest!(lc) === 5
    @test F1Tenth.take_latest!(lc) === missing
end

@testset "LatestChannel: wait_latest! blocks until a value is available" begin
    lc = F1Tenth.LatestChannel{Int}()
    t = Threads.@spawn F1Tenth.wait_latest!(lc)
    sleep(0.05)
    @test !istaskdone(t)
    F1Tenth.put_latest!(lc, 7)
    @test fetch(t) == 7
end
