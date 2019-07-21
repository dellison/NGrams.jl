using Documenter, NGrams

makedocs(sitename = "NGrams.jl",
         format = Documenter.HTML(),
         modules = [NGrams],
         pages = ["Home" => "index.md",
                  "N-Grams" => "ngrams.md",
                  "Language Modeling" => "languagemodeling.md"],
         doctest = true)

deploydocs(repo = "github.com/dellison/NGrams.jl.git")
