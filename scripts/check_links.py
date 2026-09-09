"""Check local Markdown links and anchors. External URLs are not network-tested."""
from __future__ import annotations
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def slug(text: str) -> str:
    return re.sub(r'[^\w\- ]', '', text.lower()).replace(' ', '-')


def check(root: Path = ROOT) -> int:
    failures, checked, external = [], 0, 0
    files = [p for p in root.rglob('*.md') if not any(part.startswith('.') for part in p.relative_to(root).parts)
             and not any(part in ('site-packages','node_modules','build','dist') for part in p.relative_to(root).parts)]
    for path in files:
        text = re.sub(r'```.*?```', '', path.read_text(encoding='utf-8'), flags=re.S)
        for target in re.findall(r'\[[^\]]*\]\(([^\s)]+)(?:\s+"[^"]*")?\)', text):
            parts = urlsplit(target)
            if parts.scheme or parts.netloc:
                external += 1
                continue
            dest = (path.parent / unquote(parts.path)).resolve() if parts.path else path.resolve()
            checked += 1
            if not dest.is_relative_to(root.resolve()) or not dest.exists():
                failures.append(f'{path.relative_to(root)} -> {target}')
            elif parts.fragment and dest.suffix == '.md':
                headings = re.findall(r'^#{1,6}\s+(.+?)\s*#*$', dest.read_text(encoding='utf-8'), flags=re.M)
                if unquote(parts.fragment) not in {slug(h) for h in headings}:
                    failures.append(f'{path.relative_to(root)} -> missing anchor {target}')
    if failures:
        raise ValueError('\n'.join(failures))
    print(f'Local Markdown links: {checked} passed across {len(files)} files. External URLs not tested: {external}.')
    return checked


if __name__ == '__main__':
    check()
