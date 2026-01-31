from PIL import Image, ImageDraw, ImageFont
import os

# Test text
test_text = "Utah"
font_size = 120
output_path = "font_preview.png"

# Fonts to test - (display_name, font_file, index)
fonts_to_test = [
    ("Avenir Next Bold Italic", "Avenir Next.ttc", 7),
    ("Avenir Next Ultra Light", "Avenir Next.ttc", 10),
]

# Gray shades to display (0=black, 255=white)
gray_shades = [0, 50, 100, 150, 200]

font_dir = "/System/Library/Fonts/Supplemental/"

# Calculate image size
line_height = font_size + 40
img_height = len(fonts_to_test) * len(gray_shades) * line_height + 40
img_width = 1200

# Create image
img = Image.new('RGB', (img_width, img_height), 'white')
draw = ImageDraw.Draw(img)

y = 20
for name, font_file, index in fonts_to_test:
    try:
        font_path = os.path.join(font_dir, font_file)
        font = ImageFont.truetype(font_path, font_size, index=index)
        label_font = ImageFont.truetype(font_path, 20, index=0)
        
        for gray_value in gray_shades:
            fill_color = (gray_value, gray_value, gray_value)
            
            # Draw label
            draw.text((20, y), f"{name} (gray={gray_value}):", fill='gray', font=label_font)
            
            # Draw sample text
            draw.text((450, y), test_text, fill=fill_color, font=font)
            
            y += line_height
        
    except Exception as e:
        draw.text((20, y), f"{name}: ERROR - {e}", fill='red')
        y += line_height

img.save(output_path)
print(f"Saved font preview to {output_path}")
