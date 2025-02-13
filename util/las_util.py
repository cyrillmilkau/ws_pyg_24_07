import torch
import laspy
import numpy as np

def load_las_file(las_path):
    # print(f"Read {las_path} ...")
    las = laspy.read(las_path)
    points = np.vstack((las.x, las.y, las.z)).T  # Extract XYZ coordinates
    labels = las.classification  # Extract classification labels if available
    # print(f"... done")
    return points, labels

def save_classified_las(original_las_file, predicted_labels, output_file, selected_indices=None, save_only_selected_points=False):
    '''
    Adds classification labels to a LAS file and saves a new LAS file with the labels.

    :param original_las_file: path to the input LAS file
    :param predicted_labels: a numpy array of predicted labels for each point in the point cloud
    :param output_file: the path to save the new LAS file with the classification labels
    :param selected_indices: indices of the points to update in the classification field (optional)
    '''

    # Read the original LAS file
    inFile = laspy.read(original_las_file)

    # Check if 'classification' is available in the dimensions
    classification_found = False
    for dimension in inFile.point_format.dimensions:
        if dimension.name == 'classification':
            classification_found = True
            break

    if not classification_found:
        print("Error: The LAS file format doesn't support classification.")
        return
    
    if not selected_indices:
        print("Error: The indices of points receiving the new labels are missing. Find them in PointCloudDataset return values.")
        return

    # Store the old classification values as the nex classification values before updating the labels
    new_classification_values = inFile.points.classification.copy()
    # print(f"new_classification_values: {len(new_classification_values)}")

    # Convert the tensor with new predictions to numpy and then to uint8
    predicted_labels_np = predicted_labels.cpu().to(torch.uint8).numpy()

    #--------------------------------------------------------------------------------------
    ## Dummy part --> Test arbitrary classification of classes [1] and [2]
    # num_points = len(inFile.x)
    # dummy_labels = np.zeros(num_points, dtype=np.uint8)

    # # Assign label 1 to the first half and label 2 to the second half
    # half = num_points // 2
    # dummy_labels[:half] = 1
    # dummy_labels[half:] = 2
    #--------------------------------------------------------------------------------------
    
    #--------------------------------------------------------------------------------------
    ## Selected indices part --> Apply new labels to the selected points 
    selected_indices_np = selected_indices.cpu().numpy()
    new_classification_values[selected_indices_np] = predicted_labels_np[selected_indices_np]

    #--------------------------------------------------------------------------------------

    # Add the predicted labels to the classification field
    inFile.points.classification = new_classification_values # Ensure labels are in uint8 format
    # inFile.points.classification = dummy_labels # Use the dummy labels for classification

    # # if save_only_selected_points:
    # #     # Extract the subset of points
    # #     x_subset, y_subset, z_subset, classification_subset = extract_subset_of_points(inFile, selected_indices)

    # #     # Reconstruct the LAS file with the subset
    # #     new_file = laspy.create(point_format=inFile.point_format)
    # #     new_file.x = x_subset
    # #     new_file.y = y_subset
    # #     new_file.z = z_subset
    # #     new_file.points.classification = classification_subset

    # #     # Save the updated LAS file
    # #     new_file.write(output_file)
    # #     print(f"New classified LAS file saved as: {output_file}")
    if save_only_selected_points:

        selected_indices = selected_indices.cpu().numpy()

        # Extract all point attributes for the selected points
        x_subset = inFile.x[selected_indices]
        y_subset = inFile.y[selected_indices]
        z_subset = inFile.z[selected_indices]
        classification_subset = inFile.points.classification[selected_indices]
        intensity_subset = inFile.intensity[selected_indices]
        return_number_subset = inFile.return_number[selected_indices]
        number_of_returns_subset = inFile.number_of_returns[selected_indices]
        #scan_angle_rank_subset = inFile.scan_angle_rank[selected_indices]
        #user_data_subset = inFile.user_data[selected_indices]
        #point_source_ID_subset = inFile.point_source_ID[selected_indices]
        gps_time_subset = inFile.gps_time[selected_indices]

        # Create a new LAS file with the same point format as the original
        new_file = laspy.create(point_format=inFile.point_format)

        # Add the selected points with their corresponding attributes to the new file
        new_file.x = x_subset
        new_file.y = y_subset
        new_file.z = z_subset
        new_file.points.classification = classification_subset
        new_file.intensity = intensity_subset
        new_file.return_number = return_number_subset
        new_file.number_of_returns = number_of_returns_subset
        #new_file.scan_angle_rank = scan_angle_rank_subset
        #new_file.user_data = user_data_subset
        #new_file.point_source_ID = point_source_ID_subset
        new_file.gps_time = gps_time_subset

        # Save the new LAS file
        new_file.write(output_file)
        print(f"New classified LAS-subset file saved as: {output_file}")
    else:
        # Save the full LAS file (with updated classification)
        inFile.write(output_file)
        print(f"New classified LAS file saved as: {output_file}")

# def extract_subset_of_points(inFile, selected_indices):
#     """
#     Helper function to extract a subset of points from a LAS file based on selected indices.
    
#     :param inFile: The laspy LAS file object
#     :param selected_indices: The indices of the points to extract
#     :return: A tuple (x, y, z, classification)
#     """
#     # Ensure selected indices are numpy arrays for slicing
#     selected_indices = np.array(selected_indices)
    
#     # Extract the relevant point dimensions
#     x_subset = inFile.x[selected_indices]
#     y_subset = inFile.y[selected_indices]
#     z_subset = inFile.z[selected_indices]
#     classification_subset = inFile.points.classification[selected_indices]
    
#     return x_subset, y_subset, z_subset, classification_subset