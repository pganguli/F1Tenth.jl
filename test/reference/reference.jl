# Read the gold-standard arrays written by test/reference/generate.py.
# Arrays come back in Python's index order: ref("x")[i, j] is Python's x[i-1, j-1].

const REF_DIR = joinpath(@__DIR__, "data")

const REF_SHAPES = let shapes = Dict{String,Vector{Int}}()
    for line in eachline(joinpath(REF_DIR, "manifest.txt"))
        isempty(strip(line)) && continue
        parts = split(line)
        shapes[parts[1]] = parse.(Int, parts[2:end])
    end
    shapes
end

function ref(name::AbstractString)
    dims = REF_SHAPES[name]
    raw = Vector{Float64}(undef, prod(dims))
    open(joinpath(REF_DIR, name * ".bin")) do io
        read!(io, raw)
    end
    raw .= ltoh.(raw)
    length(dims) <= 1 && return raw
    return permutedims(reshape(raw, reverse(dims)...), length(dims):-1:1)
end

const GYM_PARAMS = VehicleParams()
const EXAMPLES_DIR = joinpath(dirname(dirname(@__DIR__)), "examples")
const EXAMPLE_MAP = joinpath(EXAMPLES_DIR, "example_map")
const REF_MAPS = Dict(
    "berlin" => (joinpath(F1Tenth.MAPS_DIR, "berlin.yaml"), ".png"),
    "levine" => (joinpath(F1Tenth.MAPS_DIR, "levine.yaml"), ".pgm"),
    "skirk" => (joinpath(F1Tenth.MAPS_DIR, "skirk.yaml"), ".png"),
    "stata_basement" => (joinpath(F1Tenth.MAPS_DIR, "stata_basement.yaml"), ".png"),
    "vegas" => (joinpath(F1Tenth.MAPS_DIR, "vegas.yaml"), ".png"),
    "example_map" => (EXAMPLE_MAP * ".yaml", ".png"),
)
