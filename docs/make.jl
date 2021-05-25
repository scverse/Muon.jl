using Documenter, Muon

makedocs(sitename="Muon Documentation")

deploydocs(
    repo = "github.com/gtca/Muon.jl.git",
    devbranch = "main",
)
