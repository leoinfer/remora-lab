from __future__ import annotations

from pathlib import Path


def _points(values, width, height, pad):
    lo, hi = min(values), max(values)
    span = hi - lo or 1.0
    out = []
    for i, value in enumerate(values):
        x = pad + (width - 2 * pad) * i / max(1, len(values) - 1)
        y = height - pad - (height - 2 * pad) * (value - lo) / span
        out.append(f"{x:.2f},{y:.2f}")
    return " ".join(out)


def write_line_svg(path: str | Path, series: dict[str, list[float]], title: str = "Remora-v0 metrics") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height, pad = 900, 520, 55
    colors = ["#0b6e4f", "#b23a48", "#2455a4", "#d17a22"]
    all_values = [v for values in series.values() for v in values]
    if not all_values:
        return
    lines = []
    for idx, (name, values) in enumerate(series.items()):
        lines.append(f'<polyline fill="none" stroke="{colors[idx % len(colors)]}" stroke-width="2" points="{_points(values, width, height, pad)}"/>')
        lines.append(f'<text x="{pad + idx * 170}" y="30" fill="{colors[idx % len(colors)]}" font-size="14">{name}</text>')
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/><text x="{pad}" y="20" font-size="16">{title}</text>
<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#555"/><line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#555"/>
{''.join(lines)}
</svg>'''
    path.write_text(svg)
