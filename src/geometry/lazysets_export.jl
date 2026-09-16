import LazySets

function to_lazysets(track::Track)
    n = length(track.centerline)
    return [LazySets.VPolygon([track.outer[i], track.outer[mod1(i + 1, n)],
                track.inner[mod1(i + 1, n)], track.inner[i]])
            for i in 1:n]
end
