import sphinx_bootstrap_theme

# -- Project information -----------------------------------------------------

project = "netket"
copyright = "2019-2021, The Netket authors - All rights reserved"
author = "Giuseppe Carleo et al."

# The full version, including alpha/beta/rc tags
release = "v3.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx_reredirects",
    "sphinx_panels",
    "nbsphinx",
    "myst_parser",
    "sphinx.ext.graphviz",
    "btd.sphinx.inheritance_diagram",  # this is a custom patched version because of bug sphinx#2484
    "sphinx_rtd_theme",
]

# inheritance_graph_attrs = dict(rankdir="TB", size='""')
# graphviz_output_format = 'svg'

# Napoleon settings
autodoc_docstring_signature = True
autodoc_inherit_docstrings = True
allow_inherited = True
autosummary_generate = True
napoleon_preprocess_types = True

# PEP 526 annotations
napoleon_attr_annotations = True

panels_add_bootstrap_css = False

master_doc = "index"

autoclass_content = "class"
autodoc_class_signature = "separated"
autodoc_typehints = "description"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates", "_templates/autosummary"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# The suffix(es) of source filenames.
# Note: important to list ipynb before md here: we have both md and ipynb
# copies of each notebook, and myst will choose which to convert based on
# the order in the source_suffix list. Notebooks which are not executed have
# outputs stored in ipynb but not in md, so we must convert the ipynb.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Markdown parser latex support
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence"]
myst_update_mathjax = False
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

# -- Options for HTML output -------------------------------------------------

# html_theme = "pydata_sphinx_theme"
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
    "flax": ("https://flax.readthedocs.io/en/latest/", None),
    # "networkx": ("https://networkx.org/doc/reference/", None),
}

# (Optional) Logo. Should be small enough to fit the navbar (ideally 24x24).
# Path should be relative to the ``_static`` files directory.
html_logo = "_static/logonav.png"


# do not show __init__ if it does not have a docstring
def autodoc_skip_member(app, what, name, obj, skip, options):
    # Ref: https://stackoverflow.com/a/21449475/
    exclusions = (
        "__weakref__",  # special-members
        "__doc__",
        "__module__",
        "__dict__",  # undoc-members
    )
    exclude = name in exclusions
    if name == "__init__":
        exclude = True if obj.__doc__ is None else False
    return True if (skip or exclude) else None


## bug in sphinx: take docstring
# def warn_undocumented_members(app, what, name, obj, options, lines):
#    if name.startswith("netket"):
#        print(f"Autodoc dostuff: {what}, {name}, {obj}, {lines}, {options}")
#        print(f"the type is {type(obj)}")
#        if obj.__doc__ == None:
#
#    else:
#        print(f"Autodoc cacca: {what}, {name}, {obj}, {lines}, {options}")


def setup(app):
    app.connect("autodoc-skip-member", autodoc_skip_member)
    # app.connect('autodoc-process-docstring', warn_undocumented_members);

    # fix modules
    process_module_names(netket)
    process_module_names(netket.experimental)


import netket
import netket.experimental
import inspect


def process_module_names(module, modname="", inner=0):
    """
    This function goes through everything that is exported through __all__ in every
    module, recursively, and if it hits classes or functions it chagnes their __module__
    so that it reflects the one we want printed in the docs (instead of the actual one).

    This fixes the fact that for example netket.graph.Lattice is actually
    netket.graph.lattice.Lattice
    """
    if hasattr(module, "__all__"):
        for subm in module.__all__:
            obj = getattr(module, subm)
            process_module_names(obj, f"{module.__name__}", inner=inner + 1)
    elif inspect.isclass(module):
        module.__module__ = modname
    elif inspect.isfunction(module):
        module.__module__ = modname
