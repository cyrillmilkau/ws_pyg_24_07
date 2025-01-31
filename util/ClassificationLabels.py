from dataclasses import dataclass

@dataclass
class ClassificationLabels:
    class_names: list
    class_labels: list
    
    def __init__(self, region: str):
        """Initialize the class with a region's classification labels"""
        self.region = region
        self.class_names, self.class_labels = self.load_classification_for_region(region)
        self.__post_init__()

    def load_classification_for_region(self, region: str):
        """Load class names and labels for a specific region"""
        # You can define region-based classes and labels
        # For example, here are hardcoded labels for different regions (just for demonstration).
        if region == 'Vorarlberg':
            class_names = [
                "never classified", "unclassified", "ground", "low vegetation", 
                "medium vegetation", "high vegetation", "building", "low point", 
                "high point", "water", "bridges", "other structure", "power lines", "masts for power lines"
            ]
            class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16]
        elif region == 'Region2':  # Example for a different region
            class_names = ["class1", "class2", "class3", "class4"]
            class_labels = [0, 1, 2, 3]
        else:
            raise ValueError(f"Unknown region: {region}")

        return class_names, class_labels

    def __post_init__(self):
        """Post-initialization to prepare mappings"""
        assert len(self.class_names) == len(self.class_labels), "Number of class names does not match number of class labels!"
        self.label_to_name = dict(zip(self.class_labels, self.class_names))
        self.name_to_label = {v: k for k, v in self.label_to_name.items()}
    
    def get_label_from_name(self, name: str) -> int:
        """Get the class label by name"""
        return self.name_to_label.get(name, -1)  # Return -1 if name not found

    def get_name_from_label(self, label: int) -> str:
        """Get the class name by label"""
        return self.label_to_name.get(label, "Unknown")

    def is_valid_label(self, label: int) -> bool:
        """Check if the label is valid (within the range of defined classes)"""
        return label in self.class_labels