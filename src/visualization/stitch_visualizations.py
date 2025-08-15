# from PIL import Image, ImageDraw, ImageFont
# import os

# def compose_landscape_with_title(
#     slide_dir, slide_title,
#     map_file="prototype_map.png",
#     mixture_file="mixture_plot.png",
#     protos_file="patches.png",
#     out_file="slide_composite.png",
#     title_font_size=200,
#     bg_color=(255, 255, 255),
#     top_gutter=20,      # gap between map and mixture (top row)
#     row_gutter=50,      # gap between top row and prototypes row
#     side_gutter=20,     # left/right outer padding
#     inner_gutter=20,    # small gap between map and mixture
#     target_top_height=None  # set e.g. 1200 to force a uniform top-row height across slides
# ):
#     # --- Load images ---
#     img_map   = Image.open(os.path.join(slide_dir, map_file))
#     img_mix   = Image.open(os.path.join(slide_dir, mixture_file))
#     img_prot  = Image.open(os.path.join(slide_dir, protos_file))

#     # --- Normalize mixture: keep aspect ratio, scale to match map height (or to target_top_height) ---
#     # Determine the desired height for both top panels
#     if target_top_height is None:
#         desired_h = max(img_map.height, img_mix.height)  # use taller of the two
#     else:
#         desired_h = int(target_top_height)

#     def resize_to_height_keep_ar(img, height):
#         if img.height == height:
#             return img
#         ratio = height / float(img.height)
#         new_w = max(1, int(round(img.width * ratio)))
#         return img.resize((new_w, height), Image.LANCZOS)

#     map_top = resize_to_height_keep_ar(img_map, desired_h)
#     mix_top = resize_to_height_keep_ar(img_mix, desired_h)

#     # --- Assemble the top row: [ map | mixture ] with an inner gutter ---
#     top_row_w = map_top.width + inner_gutter + mix_top.width
#     top_row_h = desired_h

#     # --- Scale prototypes to fit full width of the top row (keep AR) ---
#     def resize_to_width_keep_ar(img, width):
#         if img.width == width:
#             return img
#         ratio = width / float(img.width)
#         new_h = max(1, int(round(img.height * ratio)))
#         return img.resize((width, new_h), Image.LANCZOS)

#     prot_bottom = resize_to_width_keep_ar(img_prot, top_row_w)

#     # --- Title bar height ---
#     title_h = title_font_size + 60  # extra padding above/below

#     # --- Final canvas size (with outer side padding) ---
#     total_w = top_row_w + 2 * side_gutter
#     total_h = title_h + top_row_h + row_gutter + prot_bottom.height + top_gutter

#     final_img = Image.new("RGB", (total_w, total_h), bg_color)

#     # --- Paste top row ---
#     x = side_gutter
#     y = title_h
#     # map (left)
#     final_img.paste(map_top, (x, y))
#     # mixture (right) with inner gutter
#     final_img.paste(mix_top, (x + map_top.width + inner_gutter, y))

#     # (optional) a small vertical gap between the two panels is already spacing horizontally via inner_gutter
#     # Add top_gutter below the top row
#     y += top_row_h + top_gutter

#     # --- Paste prototypes row (full width) ---
#     final_img.paste(prot_bottom, (x, y))
#     # no more panels under this

#     # --- Draw centered title ---
#     draw = ImageDraw.Draw(final_img)
#     try:
#         font = ImageFont.truetype("arial.ttf", title_font_size)
#     except OSError:
#         font = ImageFont.load_default()

#     # textbbox: (left, top, right, bottom)
#     try:
#         bbox = draw.textbbox((0, 0), slide_title, font=font)
#         text_w = bbox[2] - bbox[0]
#         text_h = bbox[3] - bbox[1]
#     except AttributeError:
#         # Fallback for older Pillow
#         text_w, text_h = draw.textsize(slide_title, font=font)

#     text_x = (final_img.width - text_w) // 2
#     text_y = (title_h - text_h) // 2
#     draw.text((text_x, text_y), slide_title, fill=(0, 0, 0), font=font)

#     # --- Save ---
#     out_path = os.path.join(slide_dir, out_file)
#     final_img.save(out_path, dpi=(300, 300))
#     print(f"Saved: {out_path}")


# # Example
# compose_landscape_with_title(
#     slide_dir=r"C:\Users\Vivian\Documents\PANTHER\PANTHER\features\test_slide\test_visualizations\PT 41 B",
#     slide_title="PT 41 B (5x)",
#     title_font_size=200,          # even bigger title
#     inner_gutter=24,              # small gap between map & mixture
#     row_gutter=40,                # gap before prototypes row
#     side_gutter=30,               # outer padding
#     top_gutter=24,
#     target_top_height=1200        # make the top row a consistent height across slides (optional)
# )

# ------------------------------------

from PIL import Image, ImageDraw, ImageFont
import os

def compose_landscape_with_title(
    slide_dir, slide_title,
    map_file="prototype_map.png",
    mixture_file="mixture_plot.png",
    protos_file="patches.png",
    # NEW: allow saving anywhere with a custom name
    out_path=None,                  # e.g. r"C:\out\FA 56B.png"
    out_file="slide_composite.png", # used only if out_path is None
    title_font_size=200,
    bg_color=(255, 255, 255),
    top_gutter=20,
    row_gutter=50,
    side_gutter=20,
    inner_gutter=20,
    target_top_height=None
):
    # --- Load images ---
    img_map = Image.open(os.path.join(slide_dir, map_file))
    img_mix = Image.open(os.path.join(slide_dir, mixture_file))
    img_prot = Image.open(os.path.join(slide_dir, protos_file))

    # --- Top row (same height, keep AR) ---
    desired_h = int(target_top_height) if target_top_height is not None else max(img_map.height, img_mix.height)

    def resize_to_height_keep_ar(img, h):
        if img.height == h: return img
        r = h / float(img.height)
        return img.resize((max(1, int(round(img.width * r))), h), Image.LANCZOS)

    map_top = resize_to_height_keep_ar(img_map, desired_h)
    mix_top = resize_to_height_keep_ar(img_mix, desired_h)
    top_row_w, top_row_h = map_top.width + inner_gutter + mix_top.width, desired_h

    # --- Bottom (fit width, keep AR) ---
    def resize_to_width_keep_ar(img, w):
        if img.width == w: return img
        r = w / float(img.width)
        return img.resize((w, max(1, int(round(img.height * r)))), Image.LANCZOS)

    prot_bottom = resize_to_width_keep_ar(img_prot, top_row_w)

    # --- Canvas ---
    title_h = title_font_size + 60
    total_w = top_row_w + 2 * side_gutter
    total_h = title_h + top_row_h + top_gutter + row_gutter + prot_bottom.height
    final_img = Image.new("RGB", (total_w, total_h), bg_color)

    # Paste top row
    x = side_gutter
    y = title_h
    final_img.paste(map_top, (x, y))
    final_img.paste(mix_top, (x + map_top.width + inner_gutter, y))
    y += top_row_h + top_gutter

    # Paste prototypes
    final_img.paste(prot_bottom, (x, y))

    # Title
    draw = ImageDraw.Draw(final_img)
    try:
        font = ImageFont.truetype("arial.ttf", title_font_size)
    except OSError:
        font = ImageFont.load_default()
    try:
        bbox = draw.textbbox((0, 0), slide_title, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        text_w, text_h = draw.textsize(slide_title, font=font)
    draw.text(((final_img.width - text_w)//2, (title_h - text_h)//2),
              slide_title, fill=(0,0,0), font=font)

    # Save
    if out_path is None:
        out_path = os.path.join(slide_dir, out_file)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    final_img.save(out_path, dpi=(300, 300))
    print(f"Saved: {out_path}")


def compose_all_slides(
    slides_root,                # folder containing many slide dirs
    output_dir,                 # where all composites go
    map_file="prototype_map.png",
    mixture_file="mixture_plot.png",
    protos_file="patches.png",
    title_template="{slide_id} (5x)",
    target_top_height=1200,     # keep top row consistent across slides
    skip_existing=True
):
    os.makedirs(output_dir, exist_ok=True)
    for name in sorted(os.listdir(slides_root)):
        slide_dir = os.path.join(slides_root, name)
        if not os.path.isdir(slide_dir):
            continue

        # ensure required inputs exist
        need = [os.path.join(slide_dir, map_file),
                os.path.join(slide_dir, mixture_file),
                os.path.join(slide_dir, protos_file)]
        if not all(os.path.exists(p) for p in need):
            print(f"⏭️ Missing source(s), skipping {name}")
            continue

        slide_id = name  # use directory name as slide id
        out_path = os.path.join(output_dir, f"{slide_id}.png")
        if skip_existing and os.path.exists(out_path):
            print(f"⏭️ Exists, skipping {slide_id}")
            continue

        title = title_template.format(slide_id=slide_id)
        compose_landscape_with_title(
            slide_dir=slide_dir,
            slide_title=title,
            map_file=map_file,
            mixture_file=mixture_file,
            protos_file=protos_file,
            out_path=out_path,
            title_font_size=200,
            inner_gutter=24,
            row_gutter=40,
            side_gutter=30,
            top_gutter=24,
            target_top_height=target_top_height
        )

compose_all_slides(
    slides_root=r"C:\Users\Vivian\Documents\PANTHER\PANTHER\features\test_slide\test_visualizations",
    output_dir=r"C:\Users\Vivian\Documents\PANTHER\PANTHER\features\test_slide\composites",
    map_file="prototype_map.png",
    mixture_file="mixture_plot.png",
    protos_file="patches.png",          # or "8_patches.png"
    title_template="{slide_id} (5x)",
    target_top_height=1200,
    skip_existing=True
)
