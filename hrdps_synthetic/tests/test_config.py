from pathlib import Path

import pytest
import yaml

from hrdps_synthetic import aois, config, download, generate, tile


def _yaml(tmp_path, text):
    p = tmp_path / 'settings.yaml'
    p.write_text(text)
    return p


def _parse(mod, command, argv):
    return config.parse_args(mod.build_parser(), argv, command)


def test_precedence_defaults_file_cli(tmp_path):
    cfg = _yaml(tmp_path, 'workdir: /w\nn_train: 7\ngenerate:\n  n_val: 3\n')
    a = _parse(generate, 'generate', ['--config', str(cfg)])
    assert a.workdir == Path('/w') and a.n_train == 7 and a.n_val == 3 and a.seed == 0
    a = _parse(generate, 'generate', ['--config', str(cfg), '--n-train', '9', '--workdir', '/cli'])
    assert a.n_train == 9 and a.workdir == Path('/cli') and a.n_val == 3
    a = _parse(generate, 'generate', ['--workdir', '/x'])
    assert a.n_train == 8500 and a.n_val == 1500


def test_sections_override_top_level(tmp_path):
    cfg = _yaml(tmp_path, 'workdir: /w\nworkers: 2\nchip_size: 500\ngenerate:\n  workers: 5\n')
    assert _parse(generate, 'generate', ['--config', str(cfg)]).workers == 5
    assert _parse(download, 'download', ['--config', str(cfg)]).workers == 2
    t = _parse(tile, 'tile', ['--config', str(cfg)])
    assert t.workers == 2 and t.chip_size == [500]
    assert _parse(generate, 'generate', ['--config', str(cfg)]).chip_size == [500]


def test_null_keeps_default_and_required_from_file(tmp_path):
    cfg = _yaml(tmp_path, 'workdir: /w\naoi_file: null\ngenerate:\n  workers: null\n')
    a = _parse(generate, 'generate', ['--config', str(cfg)])
    assert a.aoi_file == aois.DEFAULT_API_DUMP and a.workers >= 1
    with pytest.raises(SystemExit):
        _parse(generate, 'generate', ['--config', str(_yaml(tmp_path, 'n_train: 5\n'))])  # no workdir anywhere


@pytest.mark.parametrize('text', [
    'workdir: /w\nn_trian: 5\n',                  # typo at top level
    'workdir: /w\ngenerate:\n  chip_sise: 5\n',   # typo in a section
    'workdir: /w\ngenerate:\n  keep_grib: true\n',  # valid for download, not generate
    'workdir: /w\nchip_edge: diagonal\n',          # not a valid choice
    'workdir: /w\nn_train: lots\n',                # wrong type
    'workdir: /w\ninc_range: [20]\n',              # wrong count
    'workdir: /w\nno_scenes: yes please\n',        # not a boolean
])
def test_invalid_settings_are_rejected(tmp_path, text):
    with pytest.raises(SystemExit):
        _parse(generate, 'generate', ['--config', str(_yaml(tmp_path, text))])


def test_unquoted_times_and_days(tmp_path):
    cfg = _yaml(tmp_path, 'workdir: /w\ndownload:\n  desc_time: 14:43\n  asc_time: 02:02\n  days: [20261001, 20261003:20261005]\n')
    a = _parse(download, 'download', ['--config', str(cfg)])
    assert a.desc_time == '14:43' and a.asc_time == '02:02'
    assert a.days == ['20261001', '20261003:20261005']


def test_boolean_flags_can_be_turned_off_on_the_command_line(tmp_path):
    cfg = _yaml(tmp_path, 'workdir: /w\nchip_size: 500\ndownload:\n  keep_grib: true\ngenerate:\n  no_scenes: true\n')
    assert _parse(download, 'download', ['--config', str(cfg)]).keep_grib is True
    assert _parse(download, 'download', ['--config', str(cfg), '--no-keep-grib']).keep_grib is False
    assert _parse(generate, 'generate', ['--config', str(cfg)]).no_scenes is True
    assert _parse(generate, 'generate', ['--config', str(cfg), '--scenes']).no_scenes is False
    assert _parse(generate, 'generate', ['--workdir', '/w']).no_scenes is False
    assert _parse(generate, 'generate', ['--workdir', '/w', '--no-scenes']).no_scenes is True


def test_print_config(tmp_path, capsys):
    cfg = _yaml(tmp_path, 'workdir: /w\nwidth_km: 5\n')
    with pytest.raises(SystemExit) as e:
        _parse(generate, 'generate', ['--config', str(cfg), '--print-config', '--length-km', '6'])
    assert e.value.code == 0
    out = yaml.safe_load(capsys.readouterr().out)['generate']
    assert out['width_km'] == 5.0 and out['length_km'] == 6.0 and out['workdir'] == '/w'


def test_example_file_is_valid_and_matches_builtin_defaults():
    doc = config.load_yaml(config.EXAMPLE)
    placeholders = {('download', 'workdir'), ('generate', 'workdir'), ('tile', 'workdir'), ('rgb', 'workdir'),
                    ('tile', 'chip_size')}
    for cmd, parser in config._all_parsers().items():
        values = config.resolve(doc, cmd, parser)
        defaults = {a.dest: a.default for a in parser._actions}
        for k, v in values.items():
            if (cmd, k) in placeholders:
                continue
            d = defaults[k]
            d = list(d) if isinstance(d, tuple) else str(d) if isinstance(d, Path) else d
            assert v == d, (cmd, k, v, d)
    assert config.main(['--check', str(config.EXAMPLE)]) == 0


def test_run_saves_resolved_settings(fake_workdir, tmp_path):
    wd, aoi_list = fake_workdir
    cfg = _yaml(tmp_path, f'workdir: {wd}\naois: {[a.name for a in aoi_list]}\n'
                          'generate:\n  n_train: 4\n  n_val: 1\n  workers: 2\n  width_px: 200\n  height_px: 300\n')
    assert generate.main(['--config', str(cfg), '--limit', '1']) == 0
    saved = wd / 'settings_generate.yaml'
    assert saved.exists()
    s = yaml.safe_load(saved.read_text())['generate']
    assert s['width_px'] == 200 and s['limit'] == 1 and s['n_train'] == 4


def test_rgb_option_and_rgb_section_coexist(tmp_path):
    from hrdps_synthetic import rgb as rgb_cmd

    cfg = _yaml(tmp_path, 'workdir: /w\nrgb: false\n')
    assert _parse(generate, 'generate', ['--config', str(cfg)]).rgb is False
    cfg = _yaml(tmp_path, 'workdir: /w\nrgb:\n  sign: keep\n')
    assert _parse(rgb_cmd, 'rgb', ['--config', str(cfg)]).sign == 'keep'
    assert _parse(generate, 'generate', ['--config', str(cfg)]).rgb is True
