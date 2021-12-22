using Documenter, Muon

makedocs(sitename="Muon Documentation")

deploydocs(
    repo = "github.com/scverse/Muon.jl.git",
    devbranch = "main",
)
