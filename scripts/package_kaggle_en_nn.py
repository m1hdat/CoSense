"""Build a local, allowlisted ZIP. This command never uploads any data."""
import argparse
import json
import sys
import zipfile
from pathlib import Path

from create_splits import ensure_audit
from data_utils import ROOT, config, digest, write_json
from kaggle_en_nn import BUNDLE_VERSION, SETTINGS, load_checked, payload_paths
from validate_splits import validate


def build(output):
    cfg = config()
    ensure_audit(cfg)
    reports = {setting: validate(setting, cfg, write=False) for setting in SETTINGS}
    if not all(report['passed'] for report in reports.values()):
        raise ValueError('Split validation failed. Packaging blocked; do not regenerate membership blindly.')
    files = {relative: {'sha256': digest(ROOT / relative), 'bytes': (ROOT / relative).stat().st_size}
             for relative in payload_paths()}
    manifest = {'bundle_version': BUNDLE_VERSION, 'protocol': cfg['version'], 'scope': 'EN-NN only',
                'files': files, 'upload_status': 'not_uploaded',
                'rights_status': 'unverified; verify organizer terms before external upload',
                'final_policy': 'membership integrity only; notebook cannot evaluate final',
                'note': 'Original config/manifests retain EN-PV provenance; EN-PV TSV intentionally excluded.'}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in payload_paths():
            archive.write(ROOT / relative, relative)
        archive.writestr('bundle_manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP CRC failed.')
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        with zipfile.ZipFile(output) as archive:
            archive.extractall(temp)  # Names above are the fixed, local allowlist.
        checked = {setting: load_checked(temp, setting)[1] for setting in SETTINGS}
    verification = {'zip': output.name, 'sha256': digest(output), 'bytes': output.stat().st_size,
                    'files': len(files) + 1, 'zip_crc_passed': True,
                    'split_checks': {s: {'passed': reports[s]['passed'], 'assertions': len(reports[s]['assertions'])}
                                     for s in SETTINGS},
                    'notebook_loader_counts': {s: checked[s]['counts'] for s in SETTINGS},
                    'final_evaluated': False, 'uploaded': False}
    write_json(output.with_suffix('.verification.json'), verification)
    print(json.dumps(verification, indent=2))
    return verification


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'dist/kaggle/cosense-en-nn-data.zip')
    args = parser.parse_args()
    try:
        build(args.output)
    except (ValueError, KeyError, FileNotFoundError) as error:
        print(f'Packaging failed: {error}', file=sys.stderr)
        sys.exit(1)
