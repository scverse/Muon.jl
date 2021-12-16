using Documenter, Muon

makedocs(sitename="Muon Documentation")

deploydocs(
    repo = "github.com/PMBio/Muon.jl.git",
    devbranch = "main",
)
