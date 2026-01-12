# # Script to stitch together slide visualizations into a single composite image
# USE THIS script below

# from PIL import Image, ImageDraw, ImageFont
# import os

# def compose_landscape_with_title(
#     slide_dir, slide_title,
#     map_file="prototype_map.png",
#     mixture_file="mixture_plot.png",
#     protos_file="patches.png",
#     # NEW: allow saving anywhere with a custom name
#     out_path=None,                  # e.g. r"C:\out\FA 56B.png"
#     out_file="slide_composite.png", # used only if out_path is None
#     title_font_size=200,
#     bg_color=(255, 255, 255),
#     top_gutter=20,
#     row_gutter=50,
#     side_gutter=20,
#     inner_gutter=20,
#     target_top_height=None
# ):
#     # --- Load images ---
#     img_map = Image.open(os.path.join(slide_dir, map_file))
#     img_mix = Image.open(os.path.join(slide_dir, mixture_file))
#     img_prot = Image.open(os.path.join(slide_dir, protos_file))

#     # --- Top row (same height, keep AR) ---
#     desired_h = int(target_top_height) if target_top_height is not None else max(img_map.height, img_mix.height)

#     def resize_to_height_keep_ar(img, h):
#         if img.height == h: return img
#         r = h / float(img.height)
#         return img.resize((max(1, int(round(img.width * r))), h), Image.LANCZOS)

#     map_top = resize_to_height_keep_ar(img_map, desired_h)
#     mix_top = resize_to_height_keep_ar(img_mix, desired_h)
#     top_row_w, top_row_h = map_top.width + inner_gutter + mix_top.width, desired_h

#     # --- Bottom (fit width, keep AR) ---
#     def resize_to_width_keep_ar(img, w):
#         if img.width == w: return img
#         r = w / float(img.width)
#         return img.resize((w, max(1, int(round(img.height * r)))), Image.LANCZOS)

#     prot_bottom = resize_to_width_keep_ar(img_prot, top_row_w)

#     # --- Canvas ---
#     title_h = title_font_size + 60
#     total_w = top_row_w + 2 * side_gutter
#     total_h = title_h + top_row_h + top_gutter + row_gutter + prot_bottom.height
#     final_img = Image.new("RGB", (total_w, total_h), bg_color)

#     # Paste top row
#     x = side_gutter
#     y = title_h
#     final_img.paste(map_top, (x, y))
#     final_img.paste(mix_top, (x + map_top.width + inner_gutter, y))
#     y += top_row_h + top_gutter

#     # Paste prototypes
#     final_img.paste(prot_bottom, (x, y))

#     # Title
#     draw = ImageDraw.Draw(final_img)
#     try:
#         font = ImageFont.truetype("arial.ttf", title_font_size)
#     except OSError:
#         font = ImageFont.load_default()
#     try:
#         bbox = draw.textbbox((0, 0), slide_title, font=font)
#         text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
#     except AttributeError:
#         text_w, text_h = draw.textsize(slide_title, font=font)
#     draw.text(((final_img.width - text_w)//2, (title_h - text_h)//2),
#               slide_title, fill=(0,0,0), font=font)

#     # Save
#     if out_path is None:
#         out_path = os.path.join(slide_dir, out_file)
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     final_img.save(out_path, dpi=(300, 300))
#     print(f"Saved: {out_path}")


# def compose_all_slides(
#     slides_root,                # folder containing many slide dirs
#     output_dir,                 # where all composites go
#     map_file="prototype_map.png",
#     mixture_file="mixture_plot.png",
#     protos_file="patches.png",
#     title_template="{slide_id} (5x)",
#     target_top_height=1200,     # keep top row consistent across slides
#     skip_existing=True
# ):
#     os.makedirs(output_dir, exist_ok=True)
#     for name in sorted(os.listdir(slides_root)):
#         slide_dir = os.path.join(slides_root, name)
#         if not os.path.isdir(slide_dir):
#             continue

#         # ensure required inputs exist
#         need = [os.path.join(slide_dir, map_file),
#                 os.path.join(slide_dir, mixture_file),
#                 os.path.join(slide_dir, protos_file)]
#         if not all(os.path.exists(p) for p in need):
#             print(f"⏭️ Missing source(s), skipping {name}")
#             continue

#         slide_id = name  # use directory name as slide id
#         out_path = os.path.join(output_dir, f"{slide_id}.png")
#         if skip_existing and os.path.exists(out_path):
#             print(f"⏭️ Exists, skipping {slide_id}")
#             continue

#         title = title_template.format(slide_id=slide_id)
#         compose_landscape_with_title(
#             slide_dir=slide_dir,
#             slide_title=title,
#             map_file=map_file,
#             mixture_file=mixture_file,
#             protos_file=protos_file,
#             out_path=out_path,
#             title_font_size=200,
#             inner_gutter=24,
#             row_gutter=40,
#             side_gutter=30,
#             top_gutter=24,
#             target_top_height=target_top_height
#         )

# compose_all_slides(
#     slides_root=r"C:\Users\Vivian\Documents\PANTHER\PANTHER\features\test_slide\10x_test_visualizations",
#     output_dir=r"C:\Users\Vivian\Documents\PANTHER\PANTHER\features\test_slide\10x_composites",
#     map_file="prototype_map.png",
#     mixture_file="mixture_plot.png",
#     protos_file="patches.png",          # or "8_patches.png"
#     title_template="{slide_id} (10x)",
#     target_top_height=1200,
#     skip_existing=True
# )

# ---------------------------------------------
# compile patches from few FA and PT slides to analyze diversity of patches per protos between slides
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import os

def load_font(size):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        # Fallback to default; sizes won't be exact but it's robust
        return ImageFont.load_default()

# def resize_keep_ar_w(img, target_w):
#     if img.width == target_w: return img
#     r = target_w / float(img.width)
#     h = max(1, int(round(img.height * r)))
#     return img.resize((target_w, h), Image.LANCZOS)

def resize_keep_ar_w(img, target_w, sharpen_if_downsample=True):
    """Resize to target_w preserving AR. Never upscale. Optionally unsharp mask if downsampling."""
    src_w, src_h = img.width, img.height
    new_w = min(target_w, src_w)  # prevent upscaling
    if new_w == src_w:
        return img
    r = new_w / float(src_w)
    new_h = max(1, int(round(src_h * r)))
    out = img.resize((new_w, new_h), Image.LANCZOS)
    if sharpen_if_downsample and new_w < src_w:
        out = out.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=2))
    return out

def draw_centered_text(draw, x_center, y, text, font, fill=(0,0,0)):
    # compatibility across PIL versions
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        w, h = draw.textsize(text, font=font)
    draw.text((x_center - w//2, y), text, font=font, fill=fill)
    return h

def stack_column(slide_ids, slides_root, patches_file="patches.png",
                 target_panel_w=1600, title_font=None,
                 row_gutter=30, caption_pad=10, caption_size=38,
                 missing_ok=False):
    """
    Build a single column image by stacking 'patches.png' from each slide
    with a small centered caption under each panel.
    """
    title_font = title_font or load_font(caption_size)

    panels = []
    captions = []
    for sid in slide_ids:
        pth = os.path.join(slides_root, sid, patches_file)
        if not os.path.exists(pth):
            msg = f"[WARN] Missing {patches_file} for {sid} at {pth}"
            if missing_ok:
                print(msg)
                continue
            else:
                raise FileNotFoundError(msg)
        img = Image.open(pth).convert("RGB")
        img = resize_keep_ar_w(img, target_panel_w)
        panels.append(img)
        captions.append(sid)

    # Compute final height
    # For each panel: image height + caption height + caption_pad
    dummy_draw = ImageDraw.Draw(Image.new("RGB", (10,10)))
    caption_heights = []
    for cap in captions:
        try:
            bbox = dummy_draw.textbbox((0, 0), cap, font=title_font)
            ch = bbox[3] - bbox[1]
        except AttributeError:
            ch = dummy_draw.textsize(cap, font=title_font)[1]
        caption_heights.append(ch)

    total_h = 0
    for img, ch in zip(panels, caption_heights):
        total_h += img.height + caption_pad + ch
    total_h += row_gutter * (len(panels) - 1)

    col_img = Image.new("RGB", (target_panel_w, total_h), (255,255,255))
    y = 0
    draw = ImageDraw.Draw(col_img)

    for (img, cap, ch) in zip(panels, captions, caption_heights):
        col_img.paste(img, (0, y))
        y += img.height + caption_pad
        draw_centered_text(draw, x_center=col_img.width//2, y=y, text=cap, font=title_font)
        y += ch
        y += row_gutter

    return col_img

def compose_fa_pt_grid(fa_ids, pt_ids, slides_root,
                       patches_file="patches.png",
                       target_panel_w=1600, header_size=64,
                       col_header_pad=20, outer_pad=40,
                       col_gutter=120, row_divider=False,
                       out_path="fa_pt_composite.png"):
    """
    Create a 2-column composite:
      Left column: FA (3 slides), Right column: PT (3 slides).
    """
    header_font = load_font(header_size)
    caption_font = load_font(38)

    # Build columns (independent heights OK; we’ll align tops)
    col_fa = stack_column(fa_ids, slides_root, patches_file,
                          target_panel_w, caption_font, row_gutter=30)
    col_pt = stack_column(pt_ids, slides_root, patches_file,
                          target_panel_w, caption_font, row_gutter=30)

    # Header row height (use the tallest header text)
    tmp = Image.new("RGB", (10,10), (255,255,255))
    dtmp = ImageDraw.Draw(tmp)
    try:
        h_fa = dtmp.textbbox((0,0), "FA", font=header_font)[3]
        h_pt = dtmp.textbbox((0,0), "PT", font=header_font)[3]
    except AttributeError:
        h_fa = dtmp.textsize("FA", font=header_font)[1]
        h_pt = dtmp.textsize("PT", font=header_font)[1]
    header_h = max(h_fa, h_pt) + col_header_pad

    total_w = outer_pad + col_fa.width + col_gutter + col_pt.width + outer_pad
    total_h = outer_pad + header_h + max(col_fa.height, col_pt.height) + outer_pad
    canvas = Image.new("RGB", (total_w, total_h), (255,255,255))
    draw = ImageDraw.Draw(canvas)

    # Column x positions
    x_fa = outer_pad
    x_pt = outer_pad + col_fa.width + col_gutter

    # Draw headers centered over each column
    y_header = outer_pad
    draw_centered_text(draw, x_fa + col_fa.width//2, y_header, "FA", header_font)
    draw_centered_text(draw, x_pt + col_pt.width//2, y_header, "PT", header_font)

    # Paste columns
    y_cols = outer_pad + header_h
    canvas.paste(col_fa, (x_fa, y_cols))
    canvas.paste(col_pt, (x_pt, y_cols))

    # Optional vertical divider between columns
    if row_divider:
        mid_x = outer_pad + col_fa.width + col_gutter//2
        draw.line([(mid_x, outer_pad), (mid_x, total_h - outer_pad)], fill=(220,220,220), width=3)

    canvas.save(out_path, dpi=(300,300))
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    # ---- USER INPUTS ----
    slides_root = r"C:\Users\Vivian\Documents\PANTHER\PANTHER\features\test_slide\10x_test_visualizations"
    patches_file = "patches.png"  # change if needed (e.g., "8_patches.png")

    # Pick any 3 FA and 3 PT slide folder names (must exist under slides_root)
    fa_ids = ["FA 66 B", "FA 78 B", "FA 58B"]
    pt_ids = ["PT 35 B", "PT 39 B", "PT 52 B"]

    out_path = r"C:\Users\Vivian\Documents\PANTHER\PANTHER\features\test_slide\10x_composite_patches.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    compose_fa_pt_grid(
        fa_ids=fa_ids,
        pt_ids=pt_ids,
        slides_root=slides_root,
        patches_file=patches_file,
        target_panel_w=2800,        # adjust for slide width on your screen
        header_size=64,
        col_header_pad=20,
        outer_pad=40,
        col_gutter=120,
        row_divider=False,
        out_path=out_path
    )
