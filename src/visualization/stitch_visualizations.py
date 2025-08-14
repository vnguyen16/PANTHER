# from PIL import Image, ImageDraw, ImageFont
# import os

# def compose_with_title(slide_dir, slide_title,
#                        map_file="prototype_map.png",
#                        mixture_file="mixture_plot.png",
#                        protos_file="8_patches.png",
#                        out_file="slide_composite.png"):
#     # Load images
#     img_map = Image.open(os.path.join(slide_dir, map_file))
#     img_mix = Image.open(os.path.join(slide_dir, mixture_file))
#     img_protos = Image.open(os.path.join(slide_dir, protos_file))

#     # Resize left images to same width, align vertically
#     left_width = max(img_map.width, img_mix.width)
#     img_map = img_map.resize((left_width, img_map.height))
#     img_mix = img_mix.resize((left_width, img_mix.height))

#     # Stack map above mixture (vertically)
#     left_column = Image.new("RGB", (left_width, img_map.height + img_mix.height), (255, 255, 255))
#     left_column.paste(img_map, (0, 0))
#     left_column.paste(img_mix, (0, img_map.height))

#     # Match heights between left column and prototypes image
#     total_height = max(left_column.height, img_protos.height)
#     if left_column.height < total_height:
#         pad = Image.new("RGB", (left_column.width, total_height - left_column.height), (255, 255, 255))
#         left_column = Image.new("RGB", (left_column.width, total_height), (255, 255, 255))
#         left_column.paste(img_map, (0, 0))
#         left_column.paste(img_mix, (0, img_map.height))
#     else:
#         img_protos = img_protos.resize((img_protos.width, total_height))

#     # Combine left and right columns
#     combined = Image.new("RGB", (left_column.width + img_protos.width, total_height), (255, 255, 255))
#     combined.paste(left_column, (0, 0))
#     combined.paste(img_protos, (left_column.width, 0))

#     # Add title bar
#     title_height = 80
#     final_img = Image.new("RGB", (combined.width, combined.height + title_height), (255, 255, 255))
#     final_img.paste(combined, (0, title_height))

#     # Draw title text
#     draw = ImageDraw.Draw(final_img)
#     try:
#         font = ImageFont.truetype("arial.ttf", 48)  # Change font path/size if needed
#     except OSError:
#         font = ImageFont.load_default()
#     draw.text((20, 20), slide_title, fill=(0, 0, 0), font=font)

#     # Save output
#     out_path = os.path.join(slide_dir, out_file)
#     final_img.save(out_path, dpi=(300, 300))
#     print(f"Saved: {out_path}")

# # Example usage
# compose_with_title(
#     slide_dir=r"C:\Users\Vivian\Documents\PANTHER\PANTHER\features\test_slide\visualizations\FA_57B",
#     slide_title="FA 57B - 5x"
# )


from PIL import Image, ImageDraw, ImageFont
import os


def compose_with_title(slide_dir, slide_title,
                       map_file="prototype_map.png",
                       mixture_file="mixture_plot.png",
                       protos_file="8_patches.png",
                       out_file="slide_composite.png",
                       title_font_size=200,
                       bg_color=(255, 255, 255)):
        # Load images
    img_map = Image.open(os.path.join(slide_dir, map_file))
    img_mix = Image.open(os.path.join(slide_dir, mixture_file))
    img_protos = Image.open(os.path.join(slide_dir, protos_file))

    # Target width = max of map and mixture original widths
    target_width = max(img_map.width, img_mix.width)

    # Scale mixture to fill width but maintain aspect ratio
    mix_aspect_ratio = img_mix.height / img_mix.width
    new_mix_height = int(target_width * mix_aspect_ratio)
    img_mix = img_mix.resize((target_width, new_mix_height), Image.LANCZOS)

    # Pad map if narrower than target_width
    if img_map.width < target_width:
        padded_map = Image.new("RGB", (target_width, img_map.height), bg_color)
        offset_x = (target_width - img_map.width) // 2
        padded_map.paste(img_map, (offset_x, 0))
        img_map = padded_map

    # Stack vertically
    left_column = Image.new("RGB", (target_width, img_map.height + img_mix.height), bg_color)
    left_column.paste(img_map, (0, 0))
    left_column.paste(img_mix, (0, img_map.height))

    # Match height between left column and prototypes
    total_height = max(left_column.height, img_protos.height)
    if left_column.height < total_height:
        pad_bottom = total_height - left_column.height
        padded_left = Image.new("RGB", (left_column.width, total_height), bg_color)
        padded_left.paste(left_column, (0, 0))
        left_column = padded_left
    elif img_protos.height < total_height:
        pad_bottom = total_height - img_protos.height
        padded_protos = Image.new("RGB", (img_protos.width, total_height), bg_color)
        padded_protos.paste(img_protos, (0, 0))
        img_protos = padded_protos

    # Combine left and right columns
    combined = Image.new("RGB", (left_column.width + img_protos.width, total_height), bg_color)
    combined.paste(left_column, (0, 0))
    combined.paste(img_protos, (left_column.width, 0))

    # Add title bar
    title_height = title_font_size + 50
    final_img = Image.new("RGB", (combined.width, combined.height + title_height), bg_color)
    final_img.paste(combined, (0, title_height))

    # Draw centered title
    draw = ImageDraw.Draw(final_img)
    try:
        font = ImageFont.truetype("arial.ttf", title_font_size)
    except OSError:
        font = ImageFont.load_default()

    text_width, text_height = draw.textsize(slide_title, font=font)
    text_x = (final_img.width - text_width) // 2
    text_y = (title_height - text_height) // 2
    draw.text((text_x, text_y), slide_title, fill=(0, 0, 0), font=font)

    # Save output
    out_path = os.path.join(slide_dir, out_file)
    final_img.save(out_path, dpi=(300, 300))
    print(f"Saved: {out_path}")


# Example usage
compose_with_title(
    slide_dir=r"C:\Users\Vivian\Documents\PANTHER\PANTHER\features\test_slide\visualizations\FA_57B",
    slide_title="FA 57B (5x) Prototypes",
    title_font_size=200  # bigger title
)
