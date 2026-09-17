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
    lock::ReentrantLock
end

LatestChannel{T}() where {T} = LatestChannel{T}(Channel{T}(1), ReentrantLock())

# put_latest! and take_latest! share `lock` because each is a
# check-then-act (isready, then take!/put!) that is not atomic on its
# own -- without the lock, a concurrent put_latest!/take_latest! pair
# can race such that one side's isready check is stale by the time it
# acts, and that side's take!/put! call blocks. See the amendment note
# above this block.
function put_latest!(lc::LatestChannel{T}, v::T) where {T}
    lock(lc.lock) do
        isready(lc.ch) && take!(lc.ch)
        put!(lc.ch, v)
    end
    return nothing
end

function take_latest!(lc::LatestChannel{T}) where {T}
    return lock(lc.lock) do
        isready(lc.ch) ? take!(lc.ch) : missing
    end
end

# wait_latest! deliberately does not take `lock` -- it must be able to
# block waiting for a value without holding a lock that put_latest!
# needs in order to complete. This is safe without the lock: a put! on
# an empty Channel with a pending take! waiter is a direct,
# non-blocking handoff. This does require that wait_latest! and
# take_latest! are never both called against the same LatestChannel
# instance -- true of every use in this package (see the amendment
# note above this block).
wait_latest!(lc::LatestChannel{T}) where {T} = take!(lc.ch)
