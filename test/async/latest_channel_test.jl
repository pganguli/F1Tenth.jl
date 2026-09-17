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

@testset "LatestChannel: concurrent put_latest!/take_latest! never blocks (race regression)" begin
    lc = F1Tenth.LatestChannel{Int}()
    n = 5000
    producer_done = Threads.Atomic{Bool}(false)
    producer = Threads.@spawn begin
        for i in 1:n
            F1Tenth.put_latest!(lc, i)
        end
        producer_done[] = true
    end
    # Consumer races put_latest! with take_latest! on the same channel
    # until the producer is done, then drains any final value. Success
    # is both tasks completing within the timeout: if the pre-fix
    # check-then-act race regresses, one side's take!/put! call blocks
    # forever and this hangs until the bound trips.
    consumer = Threads.@spawn begin
        while !producer_done[]
            F1Tenth.take_latest!(lc)
            yield()
        end
        F1Tenth.take_latest!(lc)
    end
    ok = timedwait(() -> istaskdone(producer) && istaskdone(consumer), 5.0)
    @test ok === :ok
end
