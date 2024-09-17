# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "mitgcm_utils"
copyright = "2024, Prajeesh Ag"
author = "Prajeesh Ag"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


sys.path.append(os.path.abspath("../../"))

extensions = [
    "sphinx_copybutton",
    # "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx_click",
    "myst_parser",
    "sphinx_design",
    "sphinx_termynal",
]

# autodoc_mock_imports = ["f90nml", "xarray", "xesmf", "numpy", "matplotlib", "cartopy"]
autodoc_mock_imports = [
    "sphericalpolygon",
    "f90nml",
    "xarray",
    "xesmf",
    "numpy",
    "matplotlib",
    "cartopy",
    "mpl_interactions",
    "rich",
    "scipy",
    "yaml",
    "cdo",
    "pandas",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

html_theme_options = {
    "repository_url": "https://github.com/prajeeshag/mitgcm_utils",
    "use_repository_button": True,
}

html_title = "MITGCM Utils"

myst_enable_extensions = ["colon_fence"]
