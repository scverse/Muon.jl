using Documenter, DocumenterInterLinks, Muon, DataFrames

links = InterLinks(
    "Julia" => "https://docs.julialang.org/en/v1/",
    "DataFrames" => "https://dataframes.juliadata.org/stable/",
)

makedocs(
    sitename="Muon Documentation",
    warnonly=:cross_references,
    pages=["index.md", "io.md", "objects.md", "API" => ["api/types.md", "api/functions.md"]],
    plugins=[links],
)

deploydocs(repo="github.com/scverse/Muon.jl.git", devbranch="main")
