import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from light_propagation_simulation_qz import propagation

#%%
def place_square_by_index(index, array_size=100, square_size=5, distance=15):
    """
    Given an index, generate a 100x100 array and place the corresponding square region as 1,
    while the other regions remain 0. The square regions are distributed in a 3 4 3 arrangement
    and centered in the array.
    
    Parameters:
    - index: The index of the square (from 0 to 9).
    - array_size: The size of the array, default is 100x100.
    - square_size: The size of each square, default is 5x5.
    - distance: The distance between squares, default is 15.
    """
    
    # Initialize the array of size 100x100 with all values set to 0
    array = np.zeros((array_size, array_size), dtype=int)
    
    # Fixed region distribution: 3 4 3 (3 squares in the first row, 4 in the second, 3 in the third)
    arrangement = [3, 4, 3]  # 3 squares in the first row, 4 in the second, 3 in the third
    
    # Calculate the width of each row of squares
    region_width = square_size
    region_height = square_size
    row_widths = []
    for row_count in arrangement:
        total_width = row_count * region_width + (row_count - 1) * distance
        row_widths.append(total_width)
    
    # Calculate the vertical offset for the entire region (to center it vertically)
    total_height = sum([region_height] * len(arrangement)) + (len(arrangement) - 1) * distance
    start_y = (array_size - total_height) // 2  # Vertical starting position

    # Calculate the horizontal offset for the entire region (to center it horizontally)
    total_width = sum(row_widths)
    start_x = (array_size - total_width) // 2  # Horizontal starting position
    
    # Find the position of the square region corresponding to the given index
    region_count = 0
    row, col = None, None
    for r in range(len(arrangement)):  # Iterate over rows
        row_count = arrangement[r]  # Number of squares in the current row
        for c in range(row_count):  # Iterate over squares in the current row
            # If index matches, record the position
            if index == region_count:
                row, col = r, c
            region_count += 1
    
    # If the index corresponds to a valid square, calculate its position and fill the region
    if row is not None and col is not None:
        # Calculate the total width of the current row's squares
        total_width = arrangement[row] * region_width + (arrangement[row] - 1) * distance
        # Calculate the starting offset for each row (to center it horizontally)
        start_x = (array_size - total_width) // 2
        
        # Calculate the starting position of the square
        region_start_x = start_x + col * (region_width + distance)
        region_start_y = start_y + row * (region_height + distance)
        
        # Set the region of the square to 1
        array[region_start_y:region_start_y + square_size, 
              region_start_x:region_start_x + square_size] = 1

    # Visualize the array with only the square corresponding to the given index
    # plt.imshow(array, cmap='gray')
    # plt.title(f"Square in Region {index}")
    # plt.axis('off')  # Turn off axis display
    # plt.show()
    return array

# Example: Display the square in region with index 5
# index =  9   # Choose the index (0 to 9)
# sq_distance = 15
# sq_size = 7
# N_pixels = 100
# array_with_sq = place_square_by_index(index=index, array_size = N_pixels, distance=sq_distance, square_size=sq_size)
# plt.imshow(array_with_sq, cmap='gray')
# plt.title(f"Square in Region {index}")
# plt.axis('off')  # Turn off axis display
# plt.show()
#%% 

def place_square_by_index_p(index, array_size=100, square_size=5, distance=15):
    """
    Given an index, generate a 100x100 array and place the corresponding square region as 1,
    while the other regions remain 0. The square regions are distributed in a 3-4-3 arrangement
    and centered in the array.
    
    Parameters:
    - index: The index of the square (from 0 to 9).
    - array_size: The size of the array, default is 100x100.
    - square_size: The size of each square, default is 5x5.
    - distance: The distance between squares, default is 15.
    
    Returns:
    - array: A 100x100 array with the square for the given index.
    - position: A tuple (start_x, start_y) indicating the top-left corner of the square for the given index.
    """
    
    # Initialize the array of size 100x100 with all values set to 0
    array = np.zeros((array_size, array_size), dtype=int)
    
    # Fixed region distribution: 3 4 3 (3 squares in the first row, 4 in the second, 3 in the third)
    arrangement = [3, 4, 3]  # 3 squares in the first row, 4 in the second, 3 in the third
    
    # Calculate the width of each row of squares
    region_width = square_size
    region_height = square_size
    row_widths = []
    for row_count in arrangement:
        total_width = row_count * region_width + (row_count - 1) * distance
        row_widths.append(total_width)
    
    # Calculate the vertical offset for the entire region (to center it vertically)
    total_height = sum([region_height] * len(arrangement)) + (len(arrangement) - 1) * distance
    start_y = (array_size - total_height) // 2  # Vertical starting position

    # Calculate the horizontal offset for the entire region (to center it horizontally)
    total_width = sum(row_widths)
    start_x = (array_size - total_width) // 2  # Horizontal starting position
    
    # Find the position of the square region corresponding to the given index
    region_count = 0
    row, col = None, None
    for r in range(len(arrangement)):  # Iterate over rows
        row_count = arrangement[r]  # Number of squares in the current row
        for c in range(row_count):  # Iterate over squares in the current row
            # If index matches, record the position
            if index == region_count:
                row, col = r, c
            region_count += 1
    
    # If the index corresponds to a valid square, calculate its position and fill the region
    if row is not None and col is not None:
        # Calculate the total width of the current row's squares
        total_width = arrangement[row] * region_width + (arrangement[row] - 1) * distance
        # Calculate the starting offset for each row (to center it horizontally)
        start_x = (array_size - total_width) // 2
        
        # Calculate the starting position of the square
        region_start_x = start_x + col * (region_width + distance)
        region_start_y = start_y + row * (region_height + distance)
        
        # Set the region of the square to 1 in the array
        array[region_start_y:region_start_y + square_size, 
              region_start_x:region_start_x + square_size] = 1

        # Return the position of the square (top-left corner)
        position = (region_start_x, region_start_y)
    else:
        position = None  # In case the index is invalid (though it shouldn't be)

    return array, position

# Example usage:
# index = 5
# array, position = place_square_by_index_p(index)

# print(f"Position of square with index {index}: {position}")

#%%
def MNIST_lable_to_image(train_dataset,N_pixels=100,sq_distance = 15,sq_size=7):
    """
    Convert the lable to an image 
    
    Parameters:
    - train_dataset: MNIST dataset 
    - index: The index of the square (from 0 to 9).
    - array_size: The size of the array, default is 100x100.
    - square_size: The size of each square, default is 5x5.
    - distance: The distance between squares, default is 15.
    """
    arrays_with_sq = torch.zeros(N_pixels,N_pixels,10)
    
    for i in range(10):
        arrays_with_sq[:,:,i] = torch.from_numpy(place_square_by_index(index=i, array_size = N_pixels, distance=sq_distance, square_size=sq_size))
     
    # 创建用于存储转换后的图像和标签的张量
    Imgaes = torch.zeros([train_dataset.data.size(0),1,N_pixels,N_pixels],dtype=torch.double)
    Labels = torch.zeros(train_dataset.data.size(0),1,N_pixels,N_pixels,dtype=torch.double)
    print('In processing...')
    for i, (images, labels) in enumerate(train_dataset):
        Imgaes[i,0,:,:]=   images # 将输入图像存入 `Imgaes`
        
        Label_image = arrays_with_sq[:,:,labels] # 获取对应标签的图像
        Labels[i,0,:,:]=   Label_image

        # images_phase = torch.exp(1j*2.0*np.pi*images) # convert to phase image 
        # print(i)
        # if i>9999:
        #     break
    print('done')
    train_dataset_new = torch.utils.data.TensorDataset(Imgaes, Labels)
        
    return train_dataset_new

# #%%  test 
# if 0:
        
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print('Using Device: ',device)
    
#     BATCH_SIZE = 16
#     IMG_SIZE = 50
#     N_pixels = 100
#     PADDING = (N_pixels - IMG_SIZE) // 2  # zero padding, avoid loss of edge information
    
#     # Define your transformations
#     transform = transforms.Compose([
#         transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Resize to target size
#         transforms.Pad(PADDING, fill=0, padding_mode='constant'),  # Add padding with 0 (black padding)
#         transforms.ToTensor()  # Convert to tensor
#     ])
    
#     # transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((IMG_SIZE, IMG_SIZE))])
#     train_dataset = torchvision.datasets.MNIST("./data", train=True, transform=transform, download=True)
    
#     train_dataset_new = MNIST_lable_to_image(train_dataset,N_pixels=100,sq_distance = 15,sq_size=7)


# %% energy calculation 
# aa = test_output * arrays_with_sq[:,:,0]

def detector(output,detection_arrangement):
    energies = []
    
    # Iterate over each mask and calculate the energy
    for m_idx in range(10):
    
        masked_image = output.detach().cpu() * detection_arrangement[:,:,m_idx]
        
        # Calculated energy: sum of pixels in the region
        energy = torch.sum(masked_image)
        energies.append(energy)
    
        # Return the mask index corresponding to the maximum energy
    
    max_energy_index = np.argmax(energies)
    return max_energy_index

#%%
#create circles auto positions
def create_circles_auto_positions(H, W, N, radius):
    # import numpy as np
    # import matplotlib.pyplot as plt
    
    # Initialize the array with zeros
    output_image = np.zeros((H, W))
    
    # Calculate number of rows and columns in the grid based on N
    num_rows = int(np.floor(np.sqrt(N)))  # Number of rows in the grid
    num_cols = int(np.ceil(N / num_rows))  # Number of columns in the grid
    
    # Calculate horizontal and vertical spacing between circles
    row_spacing = (H - num_rows * 2 * radius) / (num_rows + 1)
    col_spacing = (W - num_cols * 2 * radius) / (num_cols + 1)
    
    # Check if there is enough space to place the circles
    if row_spacing < 0 or col_spacing < 0:
        raise ValueError('The circles cannot fit into the array with the given size and number of circles.')
    
    # Initialize counter for the number of circles placed
    circle_count = 0
    
    # Loop through rows and columns to place circles
    for r in range(1, num_rows + 1):
        for c in range(1, num_cols + 1):
            if circle_count < N:
                # Calculate the center of each circle
                center_row = round((r - 1) * (2 * radius + row_spacing) + row_spacing + radius)
                center_col = round((c - 1) * (2 * radius + col_spacing) + col_spacing + radius)
                
                # Create the circle by setting pixels within the radius to 1
                Y, X = np.ogrid[:H, :W]
                dist_from_center = np.sqrt((X - center_col) ** 2 + (Y - center_row) ** 2)
                output_image[dist_from_center <= radius] = 1
                
                # Increment the circle count
                circle_count += 1
    
    # Display the generated image
    plt.figure()
    plt.imshow(output_image, cmap='gray')
    plt.title(f'{N} circles with radius {radius}')
    plt.axis('off')
    plt.show()
    
    return output_image

#%%
# 创建自动定位的圆形区域，并返回每个区域的中心坐标
def create_detection_regions(H, W, N, radius, detectsize):
    # 初始化结果数组
    output_image = np.zeros((H, W))
    
    # 计算网格中的行和列数量
    num_rows = int(np.floor(np.sqrt(N)))  # 行数
    num_cols = int(np.ceil(N / num_rows))  # 列数
    
    # 计算圆之间的间距
    row_spacing = (H - num_rows * 2 * radius) / (num_rows + 1)
    col_spacing = (W - num_cols * 2 * radius) / (num_cols + 1)
    
    # 检查是否有足够的空间放置圆
    if row_spacing < 0 or col_spacing < 0:
        raise ValueError('给定大小和数量的圆无法放置在数组中。')
    
    # 初始化计数器和检测区域的坐标列表
    circle_count = 0
    detection_regions = []
    
    # 遍历行和列以放置圆
    for r in range(1, num_rows + 1):
        for c in range(1, num_cols + 1):
            if circle_count < N:
                # 计算每个圆的中心
                center_row = round((r - 1) * (2 * radius + row_spacing) + row_spacing + radius)
                center_col = round((c - 1) * (2 * radius + col_spacing) + col_spacing + radius)
                
                # 增加5个像素的正方形区域
                half_size = radius + detectsize
                x_start = max(center_col - half_size, 0)
                x_end = min(center_col + half_size, W)
                y_start = max(center_row - half_size, 0)
                y_end = min(center_row + half_size, H)
                detection_regions.append((x_start, x_end, y_start, y_end))
                
                # 在图像中标记圆
                Y, X = np.ogrid[:H, :W]
                dist_from_center = np.sqrt((X - center_col) ** 2 + (Y - center_row) ** 2)
                output_image[dist_from_center <= radius] = 1
                
                # 增加计数器
                circle_count += 1
    
    # 显示生成的图像
    plt.figure()
    plt.imshow(output_image, cmap='gray')
    plt.title(f'{N} Detection Regions')
    plt.axis('off')
    plt.show()
    
    return detection_regions

H, W, N, radius, detectsize = 100, 100, 3, 5, 10

# # 生成检测区域
# detection_regions = create_detection_regions(H, W, N, radius, detectsize)
# print(detection_regions)
#%%
def create_evaluation_regions(H, W, N, radius, detectsize): #以detector region圆心为中心点，detectsize为边长的正方形
    # 初始化结果数组
    output_image = np.zeros((H, W))
    
    # 计算网格中的行和列数量
    num_rows = int(np.floor(np.sqrt(N)))  # 行数
    num_cols = int(np.ceil(N / num_rows))  # 列数
    
    # 计算圆之间的间距
    row_spacing = (H - num_rows * 2 * radius) / (num_rows + 1)
    col_spacing = (W - num_cols * 2 * radius) / (num_cols + 1)
    
    # 检查是否有足够的空间放置圆
    if row_spacing < 0 or col_spacing < 0:
        raise ValueError('给定大小和数量的圆无法放置在数组中。')
    
    # 初始化计数器和检测区域的坐标列表
    circle_count = 0
    evaluation_regions = []
    
    # 遍历行和列以放置圆
    for r in range(1, num_rows + 1):
        for c in range(1, num_cols + 1):
            if circle_count < N:
                # 计算每个圆的中心
                center_row = round((r - 1) * (2 * radius + row_spacing) + row_spacing + radius)
                center_col = round((c - 1) * (2 * radius + col_spacing) + col_spacing + radius)
                
                # 增加正方形检测区域
                half_size = detectsize // 2
                x_start = max(center_col - half_size, 0)
                x_end = min(center_col + half_size, W)
                y_start = max(center_row - half_size, 0)
                y_end = min(center_row + half_size, H)
                evaluation_regions.append((x_start, x_end, y_start, y_end))
                
                # 在图像中标记圆
                Y, X = np.ogrid[:H, :W]
                dist_from_center = np.sqrt((X - center_col) ** 2 + (Y - center_row) ** 2)
                output_image[dist_from_center <= radius] = 1
                
                # 在图像中标记正方形区域
                output_image[y_start:y_end, x_start:x_end] = 0.5  # 使用灰色标记正方形区域
                
                # 增加计数器
                circle_count += 1
    
    # 显示生成的图像
    plt.figure(figsize=(8, 8))
    plt.imshow(output_image, cmap='gray')
    plt.title(f'{N} Evaluation Regions')
    plt.axis('off')
    plt.show()
    
    return evaluation_regions

# example usage
# H, W, N, radius, detectsize = 100, 100, 3, 5, 20

# # 生成检测区域
# evaluation_regions = create_evaluation_regions(H, W, N, radius, detectsize)
# print("Detection regions (x_start, x_end, y_start, y_end):")
# print(evaluation_regions)
#%%
# create the labels of the dataset
def create_labels(H, W, N, radius, Index, row_offset=0, col_offset=0):
    """
    生成单个标签图像，在图中第 Index 个位置生成圆形区域。
    当 N==1 时，直接将圆心设为图像中心加偏移，其它情况按网格排列生成圆。
    
    参数:
      H, W: 图像高度和宽度
      N: 圆形总数（网格数量）
      radius: 圆形半径
      Index: 要生成圆的序号（1~N）
      row_offset, col_offset: 对计算出的圆心在行和列方向的偏移（可正可负）
    
    返回:
      output_image: 二值图像，圆内为1，其它区域为0
    """
    output_image = np.zeros((H, W))
    
    if N == 1:
        # 直接设置圆心为图像中心，并加上偏移
        center_row = H // 2 + row_offset
        center_col = W // 2 + col_offset
    else:
        num_rows = int(np.floor(np.sqrt(N)))
        num_cols = int(np.ceil(N / num_rows))
    
        row_spacing = (H - num_rows * 2 * radius) / (num_rows + 1)
        col_spacing = (W - num_cols * 2 * radius) / (num_cols + 1)
    
        if row_spacing < 0 or col_spacing < 0:
            raise ValueError('The circles cannot fit into the array with the given size and number of circles.')
    
        circle_count = 0
        center_row = None
        center_col = None
        for r in range(1, num_rows + 1):
            for c in range(1, num_cols + 1):
                if circle_count < N:
                    circle_count += 1
                    # 计算不带偏移的圆心
                    cur_center_row = round((r - 1) * (2 * radius + row_spacing) + row_spacing + radius)
                    cur_center_col = round((c - 1) * (2 * radius + col_spacing) + col_spacing + radius)
                    if circle_count == Index:
                        center_row = cur_center_row + row_offset
                        center_col = cur_center_col + col_offset
                        break
            if center_row is not None:
                break
        if center_row is None:
            raise ValueError("Invalid Index: cannot find a corresponding circle.")
    
    Y, X = np.ogrid[:H, :W]
    dist_from_center = np.sqrt((X - center_col) ** 2 + (Y - center_row) ** 2)
    output_image[dist_from_center <= radius] = 1
    return output_image

#%%
def create_labels_4_MMF3_phase(H, W, radius, Index):
    """
    Create a label with 5 circular regions for 3-mode fiber with phase information.
    The layout is 3 circles in the top row and 2 circles in the bottom row. 
    The 4th circle is centered below the 1st and 2nd circles, and the 5th circle
    is below the 2nd and 3rd circles.

    Parameters:
        H (int): Height of the label (image) in pixels.
        W (int): Width of the label (image) in pixels.
        radius (int): Radius of the circular regions in pixels.
        Index (int): Index of the circle to set (1 to 5).

    Returns:
        output_image (ndarray): 2D array with phase information, representing the label.
    """
    # Check for valid index
    if Index < 1 or Index > 5:
        raise ValueError("Index must be between 1 and 5 (inclusive).")

    # Initialize the array with zeros
    output_image = np.zeros((H, W))

    # Calculate horizontal spacing
    top_spacing = (W - 6 * radius) / 4  # Spacing between top 3 circles

    # Row positions for top and bottom rows
    top_row_y = H // 3  # Top row is approximately 1/3 height
    bottom_row_y = 2 * H // 3  # Bottom row is approximately 2/3 height

    # Coordinates of the top row circles
    top_centers_x = [
        int((2 * radius + top_spacing) * i + radius + top_spacing) for i in range(3)
    ]

    # Coordinates for the bottom row circles
    bottom_center_x_4 = (top_centers_x[0] + top_centers_x[1]) // 2  # 4th circle
    bottom_center_x_5 = (top_centers_x[1] + top_centers_x[2]) // 2  # 5th circle

    # Top row: 3 circles
    if Index <= 3:  # If Index corresponds to a circle in the top row
        center_x = top_centers_x[Index - 1]
        center_y = top_row_y
        Y, X = np.ogrid[:H, :W]
        dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        output_image[dist_from_center <= radius] = 1
        #output_image[dist_from_center <= radius] = 0

    # Bottom row: 4th circle
    elif Index == 4:
        center_x = bottom_center_x_4
        center_y = bottom_row_y
        Y, X = np.ogrid[:H, :W]
        dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        output_image[dist_from_center <= radius] = 1 #/ (np.sqrt(5)*np.pi)#1 #2

    # Bottom row: 5th circle
    elif Index == 5:
        center_x = bottom_center_x_5
        center_y = bottom_row_y
        Y, X = np.ogrid[:H, :W]
        dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        output_image[dist_from_center <= radius] = 1 #/ (np.sqrt(5)*np.pi) ##1

    # plt.imshow(output_image, cmap='gray')
    # plt.title(f"Label for Index {Index}")
    # plt.axis('off')
    # plt.show()
    return output_image

# usage example
H = 100
W = 100
radius = 5
Index = 3
create_labels_4_MMF3_phase(H, W, radius, Index)

#%%


def create_evaluation_regions_4_MMF3_phase(H, W, radius, detectsize):
    """
    Create evaluation regions (5 regions) for a 3-mode fiber.
    The layout includes:
    - 3 circles in the top row
    - 2 circles in the bottom row:
      - The 4th circle is centered below the 1st and 2nd circles
      - The 5th circle is centered below the 2nd and 3rd circles

    Each detection region is a `detectsize × detectsize` square centered at the circle's position.

    Parameters:
        H (int): Height of the label (image) in pixels.
        W (int): Width of the label (image) in pixels.
        radius (int): Radius of the circular regions in pixels.
        detectsize (int): Side length of the square detection region.

    Returns:
        evaluation_regions (list): List of tuples containing (x_start, x_end, y_start, y_end) 
                                   for each of the 5 evaluation regions.
    """
    
    # 初始化输出图像
    output_image = np.zeros((H, W))

    # 计算水平间距
    top_spacing = (W - 6 * radius) / 4  # 上排3个圆之间的间距

    # 定义上排和下排圆的纵坐标
    top_row_y = H // 3  # 上排大约在图像1/3高度
    bottom_row_y = 2 * H // 3  # 下排大约在图像2/3高度

    # 上排3个圆的横坐标
    top_centers_x = [
        int((2 * radius + top_spacing) * i + radius + top_spacing) for i in range(3)
    ]

    # 下排两个圆的横坐标
    bottom_center_x_4 = (top_centers_x[0] + top_centers_x[1]) // 2  # 第4个圆（下方）
    bottom_center_x_5 = (top_centers_x[1] + top_centers_x[2]) // 2  # 第5个圆（下方）

    # 初始化检测区域列表
    evaluation_regions = []

    # 计算并存储所有5个区域的检测正方形坐标
    for center_x, center_y in zip(
        top_centers_x + [bottom_center_x_4, bottom_center_x_5], 
        [top_row_y] * 3 + [bottom_row_y] * 2
    ):
        half_size = detectsize // 2
        x_start = max(center_x - half_size, 0)
        x_end = min(center_x + half_size, W)
        y_start = max(center_y - half_size, 0)
        y_end = min(center_y + half_size, H)
        
        # 存储检测区域坐标
        evaluation_regions.append((x_start, x_end, y_start, y_end))

        # 标记正方形检测区域
        output_image[y_start:y_end, x_start:x_end] = 0.5  # 灰色标记正方形区域

        # 标记圆形区域
        Y, X = np.ogrid[:H, :W]
        dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        output_image[dist_from_center <= radius] = 1  # 圆形区域

    # 显示图像
    plt.figure(figsize=(8, 8))
    plt.imshow(output_image, cmap='gray')
    plt.title('Evaluation Regions for 3-Mode Fiber')
    plt.axis('off')
    plt.show()

    return evaluation_regions


# # example usage
# H = 100
# W = 100
# radius = 5
# detectsize = 10  # 正方形检测区域边长
# evaluation_regions = create_evaluation_regions_4_MMF3_phase(H, W, radius, detectsize)

# # 输出检测区域坐标
# for i, region in enumerate(evaluation_regions):
#     print(f"Region {i + 1}: x[{region[0]}:{region[1]}], y[{region[2]}:{region[3]}]")


#%%
def generate_complex_weights(num_data, num_modes,phase_option):
    #phase_option 1: The phase array will be all zeros. This will result in a phase array where the phase of all elements is 0.
    #phase_option 2: The phase values are randomly generated between 0 and 2π for all elements, except for the first column, which is set to 0.
    #phase_option 3: Similar to Option 2, but the second column is constrained to random phase values between 0 and π.
    
    # Step 1: Create a 2D array of shape (num_data, num_modes) with values in the range (0, 1)
    amplitude_raw = np.random.rand(num_data, num_modes)
    
    # Step 2: Compute the L2 norm for each row
    norms = np.linalg.norm(amplitude_raw, axis=1, keepdims=True)
    
    # Step 3: Normalize each row by dividing by its L2 norm
    amplitude = amplitude_raw / norms
    
    # phase weights generation with different setting         
    phase = np.zeros((num_data, num_modes))

    if phase_option == 1:
    # Option 1: All phases are 0--> for ODNN-based mode decomposition
        phase[:] = 0

    elif phase_option == 2:
    # Option 2: Random phase between 0 and 2π, but first column is 0
        phase[:, 1:] = np.random.uniform(0, 2*np.pi, size=(num_data, num_modes-1))
        # phase[:, 0] = 0  # Set the first column to 0

    elif phase_option == 3:
    # Option 3: Random phase between 0 and 2π, but first column is 0, second column between 0 and π
        phase[:, 1] = np.random.uniform(0, np.pi, size=num_data)  # Set second column between 0 and π    
        phase[:, 2:] = np.random.uniform(0, 2*np.pi, size=(num_data, num_modes-2))
        # phase[:, 0] = 0  # Set the first column to 0
    
    
    return amplitude, phase


#%%
# def generate_fields_ts(complex_weights,MMF_data,num_data,num_modes,image_size):
#     # could be accelerated by using tensor 
#     # MMF_data with size of (N, H, W) -> (H, W, N)
#     image_data = torch.zeros([num_data,1,image_size,image_size], dtype=torch.complex64)
#     field = torch.zeros([image_size,image_size], dtype=torch.complex64)
#     for index in range(num_data):
#         complex_weight = complex_weights[index]
#         field = (complex_weight * MMF_data).sum(dim=2)
#         image_data[index,:,:,:]=field 
            
#         # plt.figure()
#         # plt.imshow(abs(field))
#         # plt.show()
    
#     return image_data

#%%
# def generate_fields_ts(complex_weights, MMF_data, num_data, num_modes, image_size):
#     """
#     Generate field distributions using complex weights and MMF mode data.

#     Parameters:
#         complex_weights (torch.Tensor): Shape [num_data, num_modes], complex tensor.
#         MMF_data (torch.Tensor): Shape [num_modes, image_size, image_size], complex tensor.
#         num_data (int): Number of data samples.
#         num_modes (int): Number of MMF modes.
#         image_size (int): Size of the field image.

#     Returns:
#         image_data (torch.Tensor): Shape [num_data, 1, image_size, image_size], complex tensor.
#     """
#     # Ensure tensors are complex and have the correct dtype
#     # complex_weights = complex_weights.to(dtype=torch.complex64)
#     # MMF_data = MMF_data.to(dtype=torch.complex64)

#     # Initialize output tensor
#     image_data = torch.zeros([num_data, 1, image_size, image_size], dtype=torch.complex64)

#     # Compute field for each sample
#     for index in range(num_data):
#         complex_weight = complex_weights[index]  # Shape: [num_modes]
#         complex_weight = complex_weight.view(num_modes, 1, 1)  # Reshape to [num_modes, 1, 1]

#         # Element-wise multiplication and sum over modes
#         field = torch.sum(complex_weight * MMF_data, dim=0)  # Sum over `num_modes`, output [50, 50]

#         # Assign field to image_data
#         image_data[index, 0, :, :] = field  # Shape: [1, 50, 50]

#     return image_data


def generate_fields_ts(complex_weights, MMF_data, num_data, num_modes, image_size,
                       wavelength=None, z0=40e-6, dx=1e-6, device='cpu'):
    """
    Generate field distributions and propagate from fiber output to first phase screen.

    Parameters:
        complex_weights (Tensor): [num_data, num_modes], complex64.
        MMF_data       (Tensor): [num_modes, H, W], complex64 at fiber output.
        num_data (int), num_modes (int), image_size (int)
        wavelength (float): chosen lambda (m)
        z0 (float): fiber→first-screen distance (m)
        dx (float): pixel pitch (m)
        device (str): 'cpu' or 'cuda:0'

    Returns:
        image_data: [num_data,1,H,W], complex64, field on first screen.
    """
    MMF_data = MMF_data.to(device)
    image_data = torch.zeros([num_data, 1, image_size, image_size],
                             dtype=torch.complex64, device=device)

    for idx in range(num_data):
        # 1) 叠加模式
        w = complex_weights[idx].view(num_modes,1,1).to(device)
        field0 = torch.sum(w * MMF_data, dim=0)  # [H,W], at fiber output

        if wavelength is not None:
            # 2) 真实自由空间传播到第一个相位屏
            #    propagation(E, z_start, z_prop, N, dx, device, wavelength)
            field1 = propagation(field0, z0, wavelength, image_size, dx, device)
        else:
            field1 = field0

        image_data[idx,0] = field1

    return image_data
