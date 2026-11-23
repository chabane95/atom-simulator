"""Allow ``python -m atom_simulator`` as well as the installed command."""

from atom_simulator.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
