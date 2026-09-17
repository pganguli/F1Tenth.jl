# Single-producer, single-consumer "latest value" slot: put_latest!
# drops any value the consumer hasn't taken yet, so a fast producer
# never blocks and a slow consumer only ever sees the most recent
# value. Not safe for multiple concurrent producers on the same
# LatestChannel -- every use site in this package guarantees exactly
# one producer per instance (physics is the sole producer for output
# and render channels; external caller code is the sole producer for
# each agent's input channel).
struct LatestChannel{T}
    ch::Channel{T}
end

LatestChannel{T}() where {T} = LatestChannel{T}(Channel{T}(1))

# put_latest! and take_latest! each perform a check-then-act (isready,
# then take!/put!) that is not atomic on its own: without mutual
# exclusion, a concurrent pair can race such that one side's isready
# check is stale by the time it acts, and that side's take!/put! call
# blocks forever.
#
# The exclusion uses the Channel's *own* built-in ReentrantLock
# (`lock(lc.ch)`), not a separate lock, because take!/put!/isready all
# acquire that same lock internally. That matters for wait_latest!,
# which calls a bare `take!(lc.ch)` from another task: with a separate
# lock, wait_latest!'s take! could steal the pending value in the
# window between put_latest!'s isready check and its own take!, leaving
# put_latest! blocked forever on a now-empty channel while holding that
# separate lock. Holding the channel's own lock makes the compound
# operation exclude wait_latest! too.
function put_latest!(lc::LatestChannel{T}, v::T) where {T}
    lock(lc.ch) do
        isready(lc.ch) && take!(lc.ch)
        put!(lc.ch, v)
    end
    return nothing
end

function take_latest!(lc::LatestChannel{T}) where {T}
    return lock(lc.ch) do
        isready(lc.ch) ? take!(lc.ch) : missing
    end
end

# wait_latest! needs no explicit lock wrapper: take! acquires the
# channel's lock internally on its own, which is the same lock
# put_latest!/take_latest! hold for their compound operation, so it is
# correctly excluded while one of those is mid-flight and blocks until
# it completes.
wait_latest!(lc::LatestChannel{T}) where {T} = take!(lc.ch)
