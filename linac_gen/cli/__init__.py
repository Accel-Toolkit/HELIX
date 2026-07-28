"""Headless batch-mode command-line interface for HELIX.

Exposes the ``run`` / ``scan`` / ``batch`` subcommands dispatched by
``python -m linac_gen``.  All of it is GUI-free: it drives the same
``Simulation`` / ``EnvelopeSolver`` / scan-pool engines the GUI uses.
"""
