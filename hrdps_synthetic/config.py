"""YAML settings for the hrdps_synthetic commands.

Every key in the file matches a command-line option of the command that reads it.
Dashes and underscores are interchangeable (``chip-size`` == ``chip_size``).

Precedence, lowest to highest:

1. the built-in defaults (unchanged; see each command's ``--help``);
2. the YAML file given with ``--config``;
3. options typed on the command line.

Layout:

* **top-level keys** apply to every command that has that option. For example,
  ``workdir``, ``width_km`` and ``chip_size`` are shared, which keeps
  ``download``, ``generate`` and ``tile`` consistent;
* **sections** ``download:``, ``generate:`` and ``tile:`` apply to that command
  only and take precedence over top-level keys;
* a key set to ``null`` (or left out) keeps the built-in default.

``--print-config`` shows the fully resolved settings and exits. Real runs save
them to ``<workdir>/settings_<command>.yaml`` for provenance.

    python -m hrdps_synthetic.config --example > my_settings.yaml   # annotated template
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import yaml

SECTIONS = ('download', 'generate', 'tile')
_RUN_ONLY = {'help', 'config', 'print_config'}
EXAMPLE = Path(__file__).resolve().parent / 'config.example.yaml'


class ConfigError(ValueError):
    pass


# YAML 1.1 reads 14:43 as the base-60 integer 883. Use YAML 1.2-style numbers so
# times like `desc_time: 14:43` stay strings even when unquoted.
class _Loader(yaml.SafeLoader):
    pass


_Loader.yaml_implicit_resolvers = {
    k: [(tag, rx) for tag, rx in v if tag not in ('tag:yaml.org,2002:int', 'tag:yaml.org,2002:float')]
    for k, v in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_Loader.add_implicit_resolver(
    'tag:yaml.org,2002:int',
    re.compile(r'^(?:[-+]?0b[0-1_]+|[-+]?0x[0-9a-fA-F_]+|[-+]?(?:0|[1-9][0-9_]*))$'),
    list('-+0123456789'))
_Loader.add_implicit_resolver(
    'tag:yaml.org,2002:float',
    re.compile(r'^(?:[-+]?(?:[0-9][0-9_]*)?\.[0-9_]+(?:[eE][-+]?[0-9]+)?|[-+]?[0-9][0-9_]*(?:\.[0-9_]*)?[eE][-+]?[0-9]+'
               r'|[-+]?[0-9][0-9_]*\.|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$'),
    list('-+0123456789.'))


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        doc = yaml.load(f, Loader=_Loader) or {}
    if not isinstance(doc, dict):
        raise ConfigError(f'{path}: expected a mapping of settings at the top level')
    return doc


def _norm(key: Any) -> str:
    return str(key).replace('-', '_')


def _options(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    return {a.dest: a for a in parser._actions if a.dest not in _RUN_ONLY and a.option_strings}


def _all_parsers() -> dict[str, argparse.ArgumentParser]:
    from hrdps_synthetic import aois, download, generate, tile

    return {'aois': aois.build_parser(), 'download': download.build_parser(),
            'generate': generate.build_parser(), 'tile': tile.build_parser()}


def _coerce(action: argparse.Action, value: Any, where: str) -> Any:
    """Check and convert one YAML value for an argparse action."""
    name = action.dest
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse.BooleanOptionalAction)):
        if not isinstance(value, bool):
            raise ConfigError(f'{where}{name}: expected true or false, got {value!r}')
        return value

    def one(v):
        if action.type in (int, float):
            try:
                if action.type is int and isinstance(v, float) and not v.is_integer():
                    raise ValueError
                v = action.type(v)
            except (TypeError, ValueError):
                raise ConfigError(f'{where}{name}: expected {action.type.__name__}, got {v!r}') from None
        elif action.type is None and not isinstance(v, str):
            v = str(v)
        elif action.type is Path:
            v = str(v)  # argparse converts string defaults through `type`
        if action.choices is not None and v not in action.choices:
            raise ConfigError(f'{where}{name}: {v!r} is not one of {list(action.choices)}')
        return v

    if action.nargs in ('+', '*') or isinstance(action.nargs, int):
        items = value if isinstance(value, (list, tuple)) else [value]
        if isinstance(action.nargs, int) and len(items) != action.nargs:
            raise ConfigError(f'{where}{name}: expected {action.nargs} values, got {len(items)}')
        if action.nargs == '+' and not items:
            raise ConfigError(f'{where}{name}: expected at least one value')
        return [one(v) for v in items]
    if isinstance(value, (list, dict)):
        raise ConfigError(f'{where}{name}: expected a single value, got {value!r}')
    return one(value)


def resolve(doc: dict, command: str, parser: argparse.ArgumentParser) -> dict[str, Any]:
    """Settings from a YAML document that apply to `command`, validated and converted."""
    own = _options(parser)
    known_anywhere = set().union(*(_options(p) for p in _all_parsers().values()))
    top, sections = {}, {}
    for k, v in doc.items():
        if k in SECTIONS:
            if v is None:
                continue
            if not isinstance(v, dict):
                raise ConfigError(f'section {k!r} must be a mapping of settings')
            sections[k] = {_norm(kk): vv for kk, vv in v.items()}
        else:
            top[_norm(k)] = v
    unknown = sorted(k for k in top if k not in known_anywhere)
    if unknown:
        raise ConfigError(f'unknown top-level setting(s) {unknown}; see python -m hrdps_synthetic.config --example')
    for sec, vals in sections.items():
        sec_opts = _options(_all_parsers()[sec]) if sec != command else own
        bad = sorted(k for k in vals if k not in sec_opts)
        if bad:
            raise ConfigError(f'unknown setting(s) {bad} in section {sec!r}; valid: {sorted(sec_opts)}')
    merged = {k: v for k, v in top.items() if k in own}
    merged.update(sections.get(command, {}))
    out = {}
    for k, v in merged.items():
        if v is None:  # null keeps the built-in default
            continue
        where = f'{command}.' if k in sections.get(command, {}) else ''
        out[k] = _coerce(own[k], v, where)
    return out


def parse_args(parser: argparse.ArgumentParser, argv: Optional[Sequence[str]], command: str) -> argparse.Namespace:
    """Parse `argv` with built-in defaults < YAML (--config) < command line."""
    parser.add_argument('--config', type=Path, default=None,
                        help='YAML settings file (defaults < file < command line); '
                             'see python -m hrdps_synthetic.config --example')
    parser.add_argument('--print-config', action='store_true', help='print the resolved settings as YAML and exit')
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=Path, default=None)
    known, _ = pre.parse_known_args(argv)
    if known.config is not None:
        try:
            values = resolve(load_yaml(known.config), command, parser)
        except (ConfigError, yaml.YAMLError, OSError) as e:
            parser.error(f'--config {known.config}: {e}')
        actions = _options(parser)
        for k in values:
            actions[k].required = False  # satisfied by the file; the command line can still override
        parser.set_defaults(**values)
    args = parser.parse_args(argv)
    if args.print_config:
        sys.stdout.write(dump(args, command))
        parser.exit(0)
    return args


def as_dict(args: argparse.Namespace) -> dict[str, Any]:
    out = {}
    for k, v in vars(args).items():
        if k in _RUN_ONLY:
            continue
        if isinstance(v, Path):
            v = str(v)
        elif isinstance(v, tuple):
            v = list(v)
        out[k] = v
    return out


def dump(args: argparse.Namespace, command: str) -> str:
    return yaml.safe_dump({command: as_dict(args)}, sort_keys=False, default_flow_style=None)


def save_effective(args: argparse.Namespace, command: str, workdir: Path, argv: Optional[Sequence[str]] = None) -> Path:
    """Write the resolved settings of a real run to <workdir>/settings_<command>.yaml."""
    path = Path(workdir) / f'settings_{command}.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ' '.join(sys.argv if argv is None else ['python -m hrdps_synthetic.' + command, *argv])
    header = (f'# Resolved settings of the last `{command}` run, {dt.datetime.now().isoformat(timespec="seconds")}\n'
              f'# command: {cmd}\n# config file: {getattr(args, "config", None)}\n')
    path.write_text(header + dump(args, command))
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(prog='python -m hrdps_synthetic.config', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--example', action='store_true', help='print the annotated example settings file')
    p.add_argument('--check', type=Path, metavar='YAML', help='validate a settings file for every command')
    args = p.parse_args(argv)
    if args.example:
        sys.stdout.write(EXAMPLE.read_text())
        return 0
    if args.check:
        doc = load_yaml(args.check)
        for cmd, parser in _all_parsers().items():
            vals = resolve(doc, cmd, parser)
            print(f'{cmd}: {vals}')
        print('OK')
        return 0
    p.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())
