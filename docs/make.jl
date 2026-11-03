using Documenter
using F1Tenth

makedocs(;
    sitename="F1Tenth.jl",
    modules=[F1Tenth],
    pages=["Home" => "index.md", "API" => "api.md"],
    checkdocs=:exports,
    format=Documenter.HTML(; prettyurls=get(ENV, "CI", nothing) == "true"),
)

deploydocs(; repo="github.com/pganguli/F1Tenth.jl.git", devbranch="main")
