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

function put_latest!(lc::LatestChannel{T}, v::T) where {T}
    isready(lc.ch) && take!(lc.ch)
    put!(lc.ch, v)
    return nothing
end

function take_latest!(lc::LatestChannel{T}) where {T}
    return isready(lc.ch) ? take!(lc.ch) : missing
end

wait_latest!(lc::LatestChannel{T}) where {T} = take!(lc.ch)
