const _default_options = Dict(:pull_on_update => false)

"""
    set_options(; kwargs...)

Set package-specific options.

These will be written to `LocalPreferences.toml` in your active project. Supported options are:

- `pull_on_update::Bool` (defaults to `false`): Controls whether to automatically run `pull_obs!`/ `pull_var!` after `update_obs!`/`update_var!`.

See also [`get_option`](@ref).
"""
function set_options(; kwargs...)
    for (name, value) ∈ kwargs
        if !haskey(_default_options, name)
            throw(ArgumentError("unknown option $name"))
        else
            @set_preferences!(String(name) => value)
        end
    end
end

"""
    get_option(name::String)

Get package-specific options. See [`set_options`](@ref) for the list of available options.
"""
function get_option(name::String)
    try
        return @load_preference(name, _default_options[Symbol(name)])
    catch e
        if isa(e, KeyError)
            throw(ArgumentError("unknown option $name"))
        else
            rethrow()
        end
    end
end
