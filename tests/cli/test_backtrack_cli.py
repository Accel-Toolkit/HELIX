

def test_approximate_backtracking_flag_exists():
    """The CSR opt-in must be reachable from the CLI (external-review
    find: the refusal's advice was un-followable from any interface)."""
    import argparse
    from linac_gen.cli import backtrack as bt
    p = argparse.ArgumentParser()
    bt.add_arguments(p)
    args = p.parse_args(["deck.dat", "--approximate-backtracking"])
    assert args.approximate_backtracking is True
    args2 = p.parse_args(["deck.dat"])
    assert args2.approximate_backtracking is False
