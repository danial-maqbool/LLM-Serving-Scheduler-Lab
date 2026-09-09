"""Generate the allowlisted release artifacts. Git publication stays in CI YAML."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path
from llm_scheduler_lab.experiments import atomic_json, digest, read_config, run_suite, source_fingerprint
from llm_scheduler_lab.analysis import analyze
from llm_scheduler_lab.plotting import plot_all
from build_report import build
from check_artifacts import check
from check_hardware import probe
from check_links import check as check_links

ROOT = Path(__file__).resolve().parents[1]


def tooling_fingerprint() -> str:
    paths = [Path('src/llm_scheduler_lab') / name for name in ('analysis.py','statistics.py','plotting.py')]
    paths += [Path('scripts') / name for name in ('build_report.py','report_template.md.tmpl','readme_template.md.tmpl','check_artifacts.py','publish_results.py')]
    return digest({p.as_posix():hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})


def current(folder: Path) -> bool:
    try:
        marker = json.loads((folder/'publication.json').read_text(encoding='utf-8'))
        if marker['source_sha256'] != source_fingerprint() or marker['tooling_sha256'] != tooling_fingerprint():
            return False
        if marker['suite_sha256'] != digest(read_config(ROOT/'configs/study.json')):
            return False
        if not marker['files']:
            return False
        for name, expected in marker['files'].items():
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                return False
        check(folder)
        return True
    except (KeyError,ValueError,OSError,TypeError):
        return False


def main() -> None:
    folder = ROOT/'results/release'
    if current(folder):
        print('Published study fingerprints and hashes match; regeneration skipped.')
        return
    # The runner checks Git cleanliness before writing artifacts.
    run_suite(ROOT/'configs/study.json',folder)
    check(folder)
    analyze(folder,folder/'statistics')
    plot_all(folder,ROOT/'figures')
    build(folder,ROOT)
    atomic_json(folder/'hardware.json',probe())
    packages = {}
    for name in ('pytest','numpy','pandas','matplotlib','PyYAML','setuptools'):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = 'not installed'
    atomic_json(folder/'environment.json',packages)
    junit = ROOT/'results/publication-tests.xml'
    subprocess.run(['python','-m','pytest','-q',f'--junitxml={junit}'],cwd=ROOT,check=True)
    import xml.etree.ElementTree as ET
    suite = ET.parse(junit).getroot().find('testsuite')
    validation = {key:int(suite.attrib[key]) for key in ('tests','failures','errors','skipped')}
    if validation['failures'] or validation['errors']:
        raise RuntimeError('publication tests failed')
    atomic_json(folder/'validation.json',validation)
    files = [ROOT/'README.md', ROOT/'REPORT.md', ROOT/'figures/manifest.json']
    files += sorted((ROOT/'figures').glob('*.svg'))
    files += [p for p in folder.rglob('*') if p.is_file() and '.runs' not in p.parts
              and p.name != 'publication.json']
    marker = {'source_sha256':source_fingerprint(),'tooling_sha256':tooling_fingerprint(),
              'suite_sha256':digest(read_config(ROOT/'configs/study.json')),
              'files':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    atomic_json(folder/'publication.json',marker)
    check_links(ROOT)
    print('Release artifacts generated and validated. CI will commit only approved output paths.')


if __name__ == '__main__':
    main()
