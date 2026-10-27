"""PNG comparisons and an offline HTML gallery; never changes evaluation pixels."""
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .data import scale_rgb


def _font(size):
    for name in ("DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def save_comparison(source_rgb, truth, prediction, preprocessing, destination, filename, metrics):
    """Keep native panel resolution; use checkpoint fixed stretch for RGB display."""
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if truth.shape != prediction.shape or truth.ndim != 2:
        raise ValueError("Truth and prediction must be matching 2D masks")
    if source_rgb.shape != (3, *truth.shape):
        raise ValueError("RGB and mask dimensions differ")
    if not all(np.isin(a, [0, 1]).all() for a in (truth, prediction)):
        raise ValueError("Comparison requires binary 0/1 masks")
    y, p = truth.astype(bool), prediction.astype(bool)
    rgb = np.rint(scale_rgb(source_rgb, preprocessing).transpose(1, 2, 0) * 255).astype(np.uint8)
    errors = np.zeros((*y.shape, 3), dtype=np.uint8)
    errors[p & y] = (48, 194, 121)
    errors[p & ~y] = (240, 91, 100)
    errors[~p & y] = (78, 153, 244)
    panels = [Image.fromarray(rgb), Image.fromarray(y.astype(np.uint8) * 255).convert("RGB"),
              Image.fromarray(p.astype(np.uint8) * 255).convert("RGB"), Image.fromarray(errors)]
    h, w = y.shape
    gap, header, footer = 12, 96, 58
    canvas = Image.new("RGB", (w * 4 + gap * 5, h + header + footer), "#142032")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 12), f"{filename}  |  IoU {metrics['iou']:.2%}  |  F1 {metrics['f1']:.2%}",
              font=_font(24), fill="white")
    titles = ["Original RGB (fixed display stretch)", "Ground truth: white = cropland",
              "Prediction: white = cropland", "Errors: TP / FP / FN"]
    for i, (panel, title) in enumerate(zip(panels, titles)):
        left = gap + i * (w + gap)
        draw.text((left, 65), title, font=_font(19), fill="#dbe6f5")
        canvas.paste(panel, (left, header))
    draw.text((gap, header + h + 16),
              "Error map: green = correct cropland; red = false positive; blue = missed cropland; black = correct background.",
              font=_font(20), fill="#dbe6f5")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
    return destination


def write_gallery(output, report):
    """Show every image in ascending IoU order, including failures."""
    output = Path(output)
    cards = []
    for item in sorted(report["per_image"], key=lambda row: (row["iou"], row["filename"])):
        filename = escape(item["filename"])
        link = escape(item["comparison_png"], quote=True)
        cards.append(f'<article><h2>{filename} · IoU {item["iou"]:.2%} · '
                     f'Precision {item["precision"]:.2%} · Recall {item["recall"]:.2%}</h2>'
                     f'<a href="{link}"><img loading="lazy" src="{link}" alt="{filename} 对比图"></a></article>')
    text = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>耕地分割 · {escape(report['split'])} 对比图</title>
<style>body{{margin:28px auto;max-width:1600px;padding:0 20px;background:#eef2f7;color:#142032;font:16px/1.7 system-ui}}
article{{background:white;padding:16px;margin:22px 0;border-radius:12px}}img{{width:100%;height:auto}}h2{{font-size:18px}}</style>
<h1>耕地分割 · {escape(report['split'])} 全量对比图</h1>
<p>共 {report['count']} 张；全局耕地 IoU {report['metrics']['iou']:.2%}。
按单图 IoU 从低到高排序，点击图片查看原尺寸。</p>
<p>从左到右：原图、真值、预测、误差图。真值/预测中白色为耕地、黑色为背景。
误差图绿色为正确耕地，红色为误检，蓝色为漏检，黑色为正确背景。</p>
<p>原图 PNG 使用训练集确定的固定范围进行显示拉伸；原始 uint16 TIFF 未修改。
单图无真值耕地且无预测耕地时 IoU 按代码约定为 1；汇总指标由全数据像素混淆计数计算。</p>
{''.join(cards)}</html>'''
    (output / "gallery.html").write_text(text, encoding="utf-8")
