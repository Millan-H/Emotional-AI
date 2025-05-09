import random
import csv

# Define base RGB values
base_red = [255, 0, 0]  # Base red color
base_white = [255, 255, 255]  # Base white background

# Create a 28x28 list with white background and variation
rgb_array = []
for i in range(28):
    row = []
    for j in range(28):
        # Add some noise/variation to colors
        if (4 <= i <= 5 or 22 <= i <= 23) and (4 <= j <= 23):
            # Top and bottom borders with variation
            red_variation = max(0, min(255, base_red[0] + random.randint(-20, 10)))
            green_variation = max(0, min(255, base_red[1] + random.randint(0, 15)))
            blue_variation = max(0, min(255, base_red[2] + random.randint(0, 15)))
            row.append([red_variation, green_variation, blue_variation])
        elif (4 <= i <= 23) and (4 <= j <= 5 or 22 <= j <= 23):
            # Left and right borders with variation
            red_variation = max(0, min(255, base_red[0] + random.randint(-20, 10)))
            green_variation = max(0, min(255, base_red[1] + random.randint(0, 15)))
            blue_variation = max(0, min(255, base_red[2] + random.randint(0, 15)))
            row.append([red_variation, green_variation, blue_variation])
        else:
            # Background with slight noise
            white_variation_r = max(0, min(255, base_white[0] + random.randint(-10, 5)))
            white_variation_g = max(0, min(255, base_white[1] + random.randint(-10, 5)))
            white_variation_b = max(0, min(255, base_white[2] + random.randint(-10, 5)))
            row.append([white_variation_r, white_variation_g, white_variation_b])
    rgb_array.append(row)

# Add some shadow effect
for i in range(6, 24):
    for j in range(6, 24):
        if 6 <= i <= 7 and 6 <= j <= 23:
            # Shadow below top border
            current_color = rgb_array[i][j]
            rgb_array[i][j] = [max(0, c - 15) for c in current_color]
        elif 6 <= i <= 23 and 6 <= j <= 7:
            # Shadow right of left border
            current_color = rgb_array[i][j]
            rgb_array[i][j] = [max(0, c - 15) for c in current_color]

# Add some subtle texture to the red borders
for i in range(4, 24):
    for j in range(4, 24):
        if (4 <= i <= 5 or 22 <= i <= 23) and (4 <= j <= 23):
            # Texture on horizontal borders
            if random.random() < 0.3:
                current_color = rgb_array[i][j]
                rgb_array[i][j] = [max(0, current_color[0] - random.randint(10, 30)), 
                                  current_color[1], 
                                  current_color[2]]
        elif (4 <= i <= 23) and (4 <= j <= 5 or 22 <= j <= 23):
            # Texture on vertical borders
            if random.random() < 0.3:
                current_color = rgb_array[i][j]
                rgb_array[i][j] = [max(0, current_color[0] - random.randint(10, 30)), 
                                  current_color[1], 
                                  current_color[2]]

# Add a smudge in one corner
corner_i, corner_j = 22, 22
for i in range(corner_i-1, corner_i+2):
    for j in range(corner_j-1, corner_j+2):
        if 0 <= i < 28 and 0 <= j < 28:
            if rgb_array[i][j][0] > 200:  # If it's a red pixel
                # Make smudge by adding some green component
                rgb_array[i][j][1] = min(255, rgb_array[i][j][1] + random.randint(20, 40))

# Add some dust/noise pixels randomly throughout the image
for _ in range(20):
    noise_i = random.randint(0, 27)
    noise_j = random.randint(0, 27)
    noise_intensity = random.randint(10, 30)
    # Slightly darken the pixel
    rgb_array[noise_i][noise_j] = [max(0, c - noise_intensity) for c in rgb_array[noise_i][noise_j]]

# Save the RGB array as a CSV file
with open('rgb_array_28x28.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    
    # Write header row
    header = []
    for j in range(28):
        header.extend([f'R_{j}', f'G_{j}', f'B_{j}'])
    csv_writer.writerow(header)
    
    # Write data rows
    for i in range(28):
        row_data = []
        for j in range(28):
            row_data.extend(rgb_array[i][j])  # Add R,G,B values
        csv_writer.writerow(row_data)

# Also output the array as a Python list of lists for direct use
with open('rgb_array_28x28.py', 'w') as pyfile:
    pyfile.write("rgb_array = [\n")
    for row in rgb_array:
        pyfile.write("    " + str(row) + ",\n")
    pyfile.write("]\n")

print("Files generated: rgb_array_28x28.csv and rgb_array_28x28.py")