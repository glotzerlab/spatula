# Copyright (c) 2021-2026 The Regents of the University of Michigan
# Part of spatula, released under the BSD 3-Clause License.


def pytest_addoption(parser):
    g = parser.getgroup("symmetry-run")
    g.addoption(
        "--pg-first",
        action="store_true",
        help="Run only the first coordinates per PG for tests marked pg_first_only",
    )
    g.addoption(
        "--skip-boosop",
        action="store_true",
        help="Exclude parametrizations where mode == 'boosop'.",
    )


def pytest_configure(config):
    # Register the custom marker to avoid warnings
    config.addinivalue_line(
        "markers",
        "pg_first_only: limit parametrized cases to the first coordinates per PG",
    )


def pytest_collection_modifyitems(config, items):
    first_only = config.getoption("--pg-first")
    skip_boosop = config.getoption("--skip-boosop")

    chosen_shape_for_pg = {}
    deselected = []
    keep = []

    for item in list(items):
        cs = getattr(item, "callspec", None)

        # 1) Skip boosop mode if requested
        if skip_boosop and cs is not None and cs.params.get("mode") == "boosop":
            deselected.append(item)
            continue

        # 2) Keep only the first shape per point group (only for marked tests)
        if first_only and "pg_first_only" in item.keywords and cs is not None:
            params = cs.params
            if "symmetry" in params and "shape" in params:
                pg = params["symmetry"]
                if isinstance(pg, list):  # some of your tests pass ["C3"], etc.
                    pg = pg[0]
                shape_name = params["shape"]
                if pg not in chosen_shape_for_pg:
                    chosen_shape_for_pg[pg] = shape_name
                elif shape_name != chosen_shape_for_pg[pg]:
                    deselected.append(item)
                    continue

        keep.append(item)

    if deselected:
        # tell other plugins what we dropped, then update the list
        config.hook.pytest_deselected(items=deselected)
        items[:] = keep
